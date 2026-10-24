//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <c10/core/Device.h>
#include <dlpack/dlpack.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <torch/headeronly/macros/Export.h>
#include <tvm/ffi/c_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace at {
TORCH_API DLDevice torchDeviceToDLDevice(c10::Device device);
} // namespace at

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "TVMFFIToLLVMPDLLPatterns.h.inc"
#include "trident/core/Conversion/Passes.h.inc"

namespace {

mlir::Value createConstantABI(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc, int32_t typeIndex,
                              mlir::Value payload) {
  mlir::LLVM::LLVMStructType const anyTy =
      tvm_ffi::TVMFFICABIType::getLLVMType(rewriter.getContext());
  mlir::IntegerType const i32Ty = rewriter.getI32Type();
  mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
  result = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, result,
      mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, typeIndex),
      llvm::ArrayRef<int64_t>{0});
  result = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, result,
      mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0),
      llvm::ArrayRef<int64_t>{1});
  result = mlir::LLVM::InsertValueOp::create(rewriter, loc, result, payload,
                                             llvm::ArrayRef<int64_t>{2});
  return result;
}

class ConvertAsOp final : public mlir::OpConversionPattern<tvm_ffi::AsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::AsOp op, tvm_ffi::AsOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::IntegerType const i8Ty = rewriter.getI8Type();
    mlir::LLVM::LLVMStructType const tensorTy =
        dlpack::DLTensorType::getLLVMType(ctx);
    mlir::Value const tensorPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i8Ty, adaptor.getObject(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, tensorTy, tensorPtr);
    return mlir::success();
  }
};

template <typename Call>
class ConvertLocalCallOp final : public mlir::OpConversionPattern<Call> {
public:
  using mlir::OpConversionPattern<Call>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Call op, typename Call::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (llvm::any_of(op->getOperandTypes(), [](mlir::Type type) -> bool {
          return mlir::isa<tvm_ffi::ArrayType, tvm_ffi::TupleType>(type);
        })) {
      return op.emitOpError(
          "requires decompose-tvm-ffi before LLVM conversion");
    }
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Type const anyType = tvm_ffi::TVMFFICABIType::getLLVMType(context);
    mlir::Type const ptrType = mlir::LLVM::LLVMPointerType::get(context);
    mlir::Type const i32 = rewriter.getI32Type();
    mlir::Type const i64 = rewriter.getI64Type();
    std::string const name =
        llvm::formatv("__tvm_ffi_{0}", op.getCallee()).str();
    auto constant = [&](mlir::Type type, int64_t value) -> mlir::Value {
      return mlir::LLVM::ConstantOp::create(rewriter, loc, type, value);
    };
    auto allocate = [&](int64_t count) -> mlir::Value {
      return mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, anyType,
                                          constant(i64, count));
    };
    mlir::Value slots;
    mlir::Value count;
    if constexpr (std::is_same_v<Call, tvm_ffi::PackCallOp>) {
      // Forward the borrowed view without copying or owning its arguments.
      slots = mlir::LLVM::ExtractValueOp::create(rewriter, loc, ptrType,
                                                 adaptor.getArguments(),
                                                 llvm::ArrayRef<int64_t>{0});
      count = mlir::LLVM::ExtractValueOp::create(rewriter, loc, i32,
                                                 adaptor.getArguments(),
                                                 llvm::ArrayRef<int64_t>{1});
    } else {
      int64_t const size = static_cast<int64_t>(adaptor.getOperands().size());
      slots = allocate(size);
      count = constant(i32, size);
      for (auto [index, argument] : llvm::enumerate(adaptor.getOperands())) {
        mlir::Value value = argument;
        if (mlir::isa<tvm_ffi::CFunctionType>(op.getOperand(index).getType())) {
          value = createConstantABI(
              rewriter, loc, tvm_ffi::CFunctionType::getTypeIndex(),
              mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64, value));
        }
        mlir::Value const slot = mlir::LLVM::GEPOp::create(
            rewriter, loc, ptrType, anyType, slots,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
        mlir::LLVM::StoreOp::create(rewriter, loc, value, slot);
      }
    }
    mlir::Value const resultSlot = allocate(1);
    mlir::LLVM::StoreOp::create(
        rewriter, loc, mlir::LLVM::ZeroOp::create(rewriter, loc, anyType),
        resultSlot);
    mlir::Value const contextValue =
        mlir::LLVM::ZeroOp::create(rewriter, loc, ptrType);
    llvm::SmallVector<mlir::Value> const arguments{contextValue, slots, count,
                                                   resultSlot};
    mlir::Value const status =
        mlir::func::CallOp::create(rewriter, loc, name, mlir::TypeRange{i32},
                                   arguments)
            .getResult(0);
    mlir::Value const success = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::eq, status, constant(i32, 0));
    mlir::Value const flag = createConstantABI(
        rewriter, loc, tvm_ffi::BoolType::getTypeIndex(),
        mlir::LLVM::ZExtOp::create(rewriter, loc, i64, success));
    mlir::Value const none = mlir::LLVM::ZeroOp::create(rewriter, loc, anyType);
    mlir::Block *current = rewriter.getInsertionBlock();
    mlir::Block *continuation = rewriter.splitBlock(current, op->getIterator());
    mlir::Value const value = continuation->addArgument(anyType, loc);
    mlir::Block *passed =
        rewriter.createBlock(current->getParent(), continuation->getIterator());
    rewriter.setInsertionPointToEnd(current);
    mlir::LLVM::CondBrOp::create(rewriter, loc, success, passed,
                                 mlir::ValueRange{}, continuation,
                                 mlir::ValueRange{none});
    rewriter.setInsertionPointToEnd(passed);
    mlir::Value const payload =
        mlir::LLVM::LoadOp::create(rewriter, loc, anyType, resultSlot);
    mlir::LLVM::BrOp::create(rewriter, loc, mlir::ValueRange{payload},
                             continuation);
    rewriter.replaceOp(op, {flag, value});
    return mlir::success();
  }
};

class ConvertPackSizeOp final
    : public mlir::OpConversionPattern<tvm_ffi::PackSizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::PackSizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const count = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), adaptor.getArguments(),
        llvm::ArrayRef<int64_t>{1});
    rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(op, rewriter.getI64Type(),
                                                    count);
    return mlir::success();
  }
};

class ConvertPackGetOp final
    : public mlir::OpConversionPattern<tvm_ffi::PackGetOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::PackGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::Type const i64 = rewriter.getI64Type();
    mlir::Type const ptr =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Type const any =
        tvm_ffi::TVMFFICABIType::getLLVMType(rewriter.getContext());
    mlir::Value const count = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, rewriter.getI32Type(), adaptor.getArguments(),
        llvm::ArrayRef<int64_t>{1});
    mlir::Value const size =
        mlir::LLVM::SExtOp::create(rewriter, loc, i64, count);
    mlir::Value const zero =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64, 0);
    mlir::Value const nonnegative = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::sge, adaptor.getIndex(),
        zero);
    mlir::Value const below = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::slt, adaptor.getIndex(),
        size);
    mlir::Value const valid =
        mlir::LLVM::AndOp::create(rewriter, loc, nonnegative, below);
    mlir::cf::AssertOp::create(rewriter, loc, valid,
                               "packed argument index out of bounds");
    mlir::Value const data = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, ptr, adaptor.getArguments(), llvm::ArrayRef<int64_t>{0});
    mlir::Value const slot = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptr, any, data,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{adaptor.getIndex()});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, any, slot);
    return mlir::success();
  }
};

class ConvertConstantDTypeOp final
    : public mlir::OpConversionPattern<tvm_ffi::ConstantDTypeOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ConstantDTypeOp op, tvm_ffi::ConstantDTypeOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ArrayAttr const attr = op.getValue();
    const int64_t code = mlir::cast<mlir::IntegerAttr>(attr[0]).getInt() & 0xff;
    const int64_t bits = mlir::cast<mlir::IntegerAttr>(attr[1]).getInt() & 0xff;
    const int64_t lanes =
        mlir::cast<mlir::IntegerAttr>(attr[2]).getInt() & 0xffff;
    const int64_t packed = code | (bits << 8) | (lanes << 16);
    mlir::Value const payload = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(), packed);
    rewriter.replaceOp(op, createConstantABI(rewriter, op.getLoc(),
                                             tvm_ffi::DTypeType::getTypeIndex(),
                                             payload));
    return mlir::success();
  }
};

class ConvertConstantDeviceOp final
    : public mlir::OpConversionPattern<tvm_ffi::ConstantDeviceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ConstantDeviceOp op,
                  tvm_ffi::ConstantDeviceOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    c10::Device const device = op.getValue().str();
    DLDevice const dlDevice = at::torchDeviceToDLDevice(device);
    mlir::LLVM::LLVMStructType const dlDeviceTy =
        dlpack::DLDeviceType::getLLVMType(rewriter.getContext());
    mlir::Value deviceValue =
        mlir::LLVM::UndefOp::create(rewriter, loc, dlDeviceTy);
    deviceValue = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, deviceValue,
        mlir::LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI32Type(),
            static_cast<int32_t>(dlDevice.device_type)),
        llvm::ArrayRef<int64_t>{0});
    deviceValue = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, deviceValue,
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       dlDevice.device_id),
        llvm::ArrayRef<int64_t>{1});
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value const deviceSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, dlDeviceTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
    mlir::LLVM::StoreOp::create(rewriter, loc, deviceValue, deviceSlot);
    mlir::Value const payload =
        mlir::LLVM::LoadOp::create(rewriter, loc, i64Ty, deviceSlot);
    rewriter.replaceOp(
        op, createConstantABI(rewriter, loc,
                              tvm_ffi::DeviceType::getTypeIndex(), payload));
    return mlir::success();
  }
};

class ConvertConstantRawStrOp final
    : public mlir::OpConversionPattern<tvm_ffi::ConstantRawStrOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ConstantRawStrOp op,
                  tvm_ffi::ConstantRawStrOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const stringPtr =
        utils::getString(rewriter, op.getLoc(), op.getValue());
    mlir::Value const payload = mlir::LLVM::PtrToIntOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(), stringPtr);
    rewriter.replaceOp(
        op, createConstantABI(rewriter, op.getLoc(),
                              tvm_ffi::RawStrType::getTypeIndex(), payload));
    return mlir::success();
  }
};

class ConvertEqOp final : public mlir::OpConversionPattern<tvm_ffi::EqOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::EqOp op, tvm_ffi::EqOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::IntegerType const i32Ty = rewriter.getI32Type();
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }

    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFICABIType::getLLVMType(rewriter.getContext());
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
    mlir::Value const lhsStorage =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::Value const rhsStorage =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getLhs(), lhsStorage);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getRhs(), rhsStorage);
    mlir::Value const mapFreeVars = tvm_ffi::IntType::build(rewriter, loc, 0);
    mlir::Value const skipTensorContent =
        tvm_ffi::IntType::build(rewriter, loc, 0);
    mlir::FailureOr<mlir::Value> result =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "ffi.StructuralEqual",
            {lhsStorage, rhsStorage, mapFreeVars, skipTensorContent});
    if (mlir::failed(result)) {
      return op.emitError("failed to call ffi.StructuralEqual");
    }
    mlir::Value const resultPayloadI64 =
        tvm_ffi::TVMFFIABIType::load(rewriter, loc, result.value());
    mlir::Value const resultPayload =
        mlir::LLVM::TruncOp::create(rewriter, loc, i32Ty, resultPayloadI64);
    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
        op, mlir::arith::CmpIPredicate::ne, resultPayload,
        mlir::arith::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(0)));
    return mlir::success();
  }
};

class ConvertExceptionOp final
    : public mlir::OpConversionPattern<tvm_ffi::ExceptionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ExceptionOp op, tvm_ffi::ExceptionOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Location const loc = op.getLoc();
    mlir::IntegerType const i32Ty = rewriter.getIntegerType(32);
    mlir::IntegerType const i64Ty = rewriter.getIntegerType(64);
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFICABIType::getLLVMType(context);
    mlir::Value const kindPtr = utils::getString(rewriter, loc, op.getKind());
    mlir::Value exceptionArg =
        mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    exceptionArg = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, exceptionArg,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 8),
        llvm::ArrayRef<int64_t>{0});
    exceptionArg = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, exceptionArg,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0),
        llvm::ArrayRef<int64_t>{1});
    exceptionArg = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, exceptionArg,
        mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, kindPtr),
        llvm::ArrayRef<int64_t>{2});
    mlir::Value const kindSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
    mlir::LLVM::StoreOp::create(rewriter, loc, exceptionArg, kindSlot);
    mlir::FailureOr<mlir::Value> resultSlot =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.ffi.Exception",
            llvm::ArrayRef<mlir::Value>{kindSlot});
    if (mlir::failed(resultSlot)) {
      return op.emitError("failed to create TVM FFI exception");
    }
    mlir::Value const exception =
        mlir::LLVM::LoadOp::create(rewriter, loc, anyTy, resultSlot.value())
            .getResult();
    rewriter.replaceOp(op, exception);
    return mlir::success();
  }
};

class ConvertFunctionCallOp final
    : public mlir::OpConversionPattern<tvm_ffi::FunctionCallOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::FunctionCallOp op,
                  tvm_ffi::FunctionCallOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const nonnull = mlir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::ne,
        adaptor.getCallee(),
        mlir::LLVM::ZeroOp::create(
            rewriter, op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(rewriter.getContext())));
    if (mlir::failed(utils::emitLLVMAssert(rewriter, op.getLoc(), nonnull))) {
      return op.emitError("failed to assert non-null function handle");
    }
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    mlir::LLVM::LLVMStructType anyTy =
        tvm_ffi::TVMFFICABIType::getLLVMType(rewriter.getContext());
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    mlir::Value const count = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Ty,
        static_cast<int64_t>(adaptor.getArguments().size()));
    mlir::Value slots = mlir::LLVM::AllocaOp::create(rewriter, op.getLoc(),
                                                     ptrTy, anyTy, count);
    llvm::SmallVector<mlir::Value> const slotPtrs = llvm::map_to_vector(
        llvm::enumerate(adaptor.getArguments()),
        [&](auto indexedArgument) -> mlir::Value {
          auto [i, argument] = indexedArgument;
          mlir::Value slot = mlir::LLVM::GEPOp::create(
              rewriter, op.getLoc(), ptrTy, anyTy, slots,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
          mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), argument, slot);
          return slot;
        });
    mlir::Value const resultSlot = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 1));
    mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(),
        mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), anyTy), resultSlot);
    mlir::FailureOr<mlir::Value> status = conversion::utils::callTVMFFIFunction(
        rewriter, op.getLoc(), module, adaptor.getCallee(), slotPtrs,
        resultSlot);
    if (mlir::failed(status)) {
      return op.emitError("failed to lower TVM FFI function call");
    }
    mlir::Value const zero = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), 0);
    mlir::Value const success = mlir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::eq, status.value(),
        zero);
    if (mlir::failed(utils::emitLLVMAssert(rewriter, op.getLoc(), success))) {
      return op.emitError("failed to assert function call success");
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, anyTy, resultSlot);
    return mlir::success();
  }
};

class ConvertFunctionGetGlobalOp final
    : public mlir::OpConversionPattern<tvm_ffi::FunctionGetGlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::FunctionGetGlobalOp op,
                  tvm_ffi::FunctionGetGlobalOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    mlir::FailureOr<std::tuple<mlir::Value, mlir::Value>> function =
        conversion::utils::getTVMFFIGlobalFunction(rewriter, op.getLoc(),
                                                   module, op.getName());
    if (mlir::failed(function)) {
      return op.emitError("failed to get TVM FFI global function");
    }
    auto [handle, status] = *function;
    mlir::Value const zero = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), 0);
    mlir::Value const success = mlir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::eq, status, zero);
    if (mlir::failed(utils::emitLLVMAssert(rewriter, op.getLoc(), success))) {
      return op.emitError("failed to assert global lookup success");
    }
    mlir::Value const nonnull = mlir::LLVM::ICmpOp::create(
        rewriter, op.getLoc(), mlir::LLVM::ICmpPredicate::ne, handle,
        mlir::LLVM::ZeroOp::create(
            rewriter, op.getLoc(),
            mlir::LLVM::LLVMPointerType::get(rewriter.getContext())));
    if (mlir::failed(utils::emitLLVMAssert(rewriter, op.getLoc(), nonnull))) {
      return op.emitError("failed to assert non-null global handle");
    }
    rewriter.replaceOp(op, handle);
    return mlir::success();
  }
};

class ConvertGetOp final : public mlir::OpConversionPattern<tvm_ffi::GetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::GetOp op, tvm_ffi::GetOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type const operandType = op.getOperand().getType();
    mlir::Type const resultType = op.getResult().getType();

    mlir::Value const payload = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, rewriter.getI64Type(), adaptor.getOperand(),
        llvm::ArrayRef<int64_t>{2});

    if (mlir::isa<tvm_ffi::BoolType>(operandType) &&
        resultType.isSignlessInteger(1)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(op, resultType, payload);
    } else if (mlir::isa<tvm_ffi::FloatType>(operandType) &&
               mlir::isa<mlir::FloatType>(resultType)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(op, resultType,
                                                         payload);
    } else if (mlir::isa<tvm_ffi::IntType>(operandType) &&
               mlir::isa<mlir::IntegerType>(resultType)) {
      rewriter.replaceOp(op, payload);
    } else if (mlir::isa<tvm_ffi::TVMFFIObjectType>(operandType) &&
               mlir::isa<tvm_ffi::ObjectType>(resultType)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(
          op, mlir::LLVM::LLVMPointerType::get(ctx), payload);
    } else {
      return op.emitOpError("unsupported get from ")
             << operandType << " to " << resultType;
    }
    return mlir::success();
  }
};

template <typename Op, auto GetCallee>
class ConvertRefOp final : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> callee = GetCallee(module);
    if (mlir::failed(callee)) {
      return op.emitError("failed to declare TVM FFI reference operation");
    }

    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value handle = adaptor.getObject();
    if (!mlir::isa<tvm_ffi::CFunctionType, tvm_ffi::ObjectType>(
            op.getObject().getType())) {
      mlir::Value const payload = mlir::LLVM::ExtractValueOp::create(
          rewriter, loc, adaptor.getObject(), llvm::ArrayRef<int64_t>{2});
      handle = mlir::LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, payload);
    }

    // A !tvm_ffi.c.any value can contain a scalar, so inspect its ABI
    // TypeIndex before passing the payload to ObjectIncRef/ObjectDecRef.
    // Statically typed TVM FFI objects retain the direct call path.
    mlir::Value ready = mlir::LLVM::ICmpOp::create(
        rewriter, loc, mlir::LLVM::ICmpPredicate::ne, handle,
        mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy));
    if (!mlir::isa<tvm_ffi::ObjectType, tvm_ffi::CFunctionType>(
            op.getObject().getType())) {
      mlir::Value const typeIndex = mlir::LLVM::ExtractValueOp::create(
          rewriter, loc, rewriter.getI32Type(), adaptor.getObject(),
          llvm::ArrayRef<int64_t>{0});
      mlir::Value const isObject = mlir::LLVM::ICmpOp::create(
          rewriter, loc, mlir::LLVM::ICmpPredicate::uge, typeIndex,
          mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                         kTVMFFIStaticObjectBegin));
      ready = mlir::LLVM::AndOp::create(rewriter, loc, ready, isObject);
    }
    mlir::Block *currentBlock = rewriter.getInsertionBlock();
    mlir::Block *continuation =
        rewriter.splitBlock(currentBlock, op->getIterator());
    mlir::Block *callBlock = rewriter.createBlock(currentBlock->getParent(),
                                                  continuation->getIterator());
    rewriter.setInsertionPointToEnd(currentBlock);
    mlir::LLVM::CondBrOp::create(rewriter, loc, ready, callBlock, continuation);
    rewriter.setInsertionPointToEnd(callBlock);
    mlir::LLVM::CallOp::create(rewriter, loc, callee.value(),
                               mlir::ValueRange{handle});
    mlir::LLVM::BrOp::create(rewriter, loc, continuation);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertIsaOp final : public mlir::OpConversionPattern<tvm_ffi::IsaOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::IsaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    tvm_ffi::TVMFFITypeIndexInterface const indexed =
        mlir::dyn_cast<tvm_ffi::TVMFFITypeIndexInterface>(op.getTarget());
    if (!indexed) {
      return op.emitOpError(
          "requires decompose-tvm-ffi before LLVM conversion");
    }
    mlir::Value const tag = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), adaptor.getValue(),
        llvm::ArrayRef<int64_t>{0});
    mlir::Value const expected = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), indexed.getTypeIndex());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        op, mlir::LLVM::ICmpPredicate::eq, tag, expected);
    return mlir::success();
  }
};

class ConvertTVMFFIDowncastOp final
    : public mlir::OpConversionPattern<tvm_ffi::DowncastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::DowncastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getValue().getType() == op.getType()) {
      rewriter.replaceOp(op, adaptor.getValue());
      return mlir::success();
    }
    mlir::Location const loc = op.getLoc();
    mlir::Value const tag = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, rewriter.getI32Type(), adaptor.getValue(),
        llvm::ArrayRef<int64_t>{0});
    llvm::SmallVector<mlir::Type> const types =
        llvm::TypeSwitch<mlir::Type, llvm::SmallVector<mlir::Type>>(
            op.getType())
            .Case<tvm_ffi::UnionType>(
                [](tvm_ffi::UnionType type) -> llvm::SmallVector<mlir::Type> {
                  return llvm::SmallVector<mlir::Type>(type.getTypes().begin(),
                                                       type.getTypes().end());
                })
            .Default([](mlir::Type type) -> llvm::SmallVector<mlir::Type> {
              return {type};
            });
    mlir::Value condition = mlir::LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI1Type(), false);
    for (mlir::Type const type : types) {
      tvm_ffi::TVMFFITypeIndexInterface const indexed =
          mlir::dyn_cast<tvm_ffi::TVMFFITypeIndexInterface>(type);
      if (!indexed) {
        return op.emitOpError(
            "requires decompose-tvm-ffi before LLVM conversion");
      }
      int32_t const index = indexed.getTypeIndex();
      mlir::Value const expected = mlir::LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), index);
      mlir::Value const matches = mlir::LLVM::ICmpOp::create(
          rewriter, loc, mlir::LLVM::ICmpPredicate::eq, tag, expected);
      condition = mlir::LLVM::OrOp::create(rewriter, loc, condition, matches);
    }
    if (mlir::failed(utils::emitLLVMAssert(rewriter, loc, condition))) {
      return op.emitError("failed to emit LLVM assertion");
    }
    mlir::Value value = adaptor.getValue();
    if (mlir::isa<tvm_ffi::CFunctionType>(op.getType())) {
      mlir::Value const payload = mlir::LLVM::ExtractValueOp::create(
          rewriter, loc, rewriter.getI64Type(), value,
          llvm::ArrayRef<int64_t>{2});
      value = mlir::LLVM::IntToPtrOp::create(
          rewriter, loc,
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), payload);
    }
    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

class ConvertTVMFFIToLLVMPass final
    : public impl::ConvertTVMFFIToLLVMBase<ConvertTVMFFIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);

    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);
    target
        .addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect,
                         mlir::arith::ArithDialect,
                         mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                         mlir::gpu::GPUDialect, mlir::scf::SCFDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) -> bool {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) -> bool {
          return typeConverter.isLegal(op);
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) -> bool {
          return mlir::isLegalForReturnOpTypeConversionPattern(op,
                                                               typeConverter);
        });
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

class ConvertTVMFFIUpcastOp final
    : public mlir::OpConversionPattern<tvm_ffi::UpcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::UpcastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value value = adaptor.getValue();
    if (mlir::isa<tvm_ffi::CFunctionType>(op.getValue().getType()) &&
        op.getValue().getType() != op.getType()) {
      mlir::Value const payload = mlir::LLVM::PtrToIntOp::create(
          rewriter, op.getLoc(), rewriter.getI64Type(), value);
      value =
          createConstantABI(rewriter, op.getLoc(),
                            tvm_ffi::CFunctionType::getTypeIndex(), payload);
    }
    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

class ConvertTensorCloneOp final
    : public mlir::OpConversionPattern<tvm_ffi::TensorCloneOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorCloneOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }
    mlir::Location const loc = op.getLoc();
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFICABIType::getLLVMType(rewriter.getContext());
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 1);
    mlir::Value const sourceSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getTensor(), sourceSlot);
    mlir::FailureOr<mlir::Value> resultSlot =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.runtime.tensor_clone",
            {sourceSlot});
    if (mlir::failed(resultSlot)) {
      return op.emitError("failed to call TVM FFI tensor clone helper");
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, anyTy,
                                                    resultSlot.value());
    return mlir::success();
  }
};

class ConvertTensorCopyOp final
    : public mlir::OpConversionPattern<tvm_ffi::TensorCopyOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorCopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }
    mlir::Location const loc = op.getLoc();
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFICABIType::getLLVMType(rewriter.getContext());
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 1);
    mlir::Value const destinationSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::Value const sourceSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getDestination(),
                                destinationSlot);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getSource(), sourceSlot);
    if (mlir::failed(conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.runtime.tensor_copy_",
            {destinationSlot, sourceSlot}))) {
      return op.emitError("failed to call TVM FFI tensor copy helper");
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertTensorLiteralOp final
    : public mlir::OpConversionPattern<tvm_ffi::TensorLiteralOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorLiteralOp op, tvm_ffi::TensorLiteralOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::DenseElementsAttr const dense =
        mlir::cast<mlir::DenseElementsAttr>(op.getValue());
    mlir::RankedTensorType const denseType =
        mlir::cast<mlir::RankedTensorType>(dense.getType());
    mlir::Location loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::LLVM::LLVMPointerType ptrType =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::IntegerType const i8Type = rewriter.getI8Type();
    mlir::IntegerType i32Type = rewriter.getI32Type();
    mlir::IntegerType const i64Type = rewriter.getI64Type();

    int64_t elementBits;
    int32_t dtypeCode;
    if (mlir::FloatType floatType =
            mlir::dyn_cast<mlir::FloatType>(denseType.getElementType())) {
      elementBits = floatType.getWidth();
      dtypeCode = kDLFloat;
    } else if (mlir::IntegerType const integerType =
                   mlir::dyn_cast<mlir::IntegerType>(
                       denseType.getElementType())) {
      elementBits =
          integerType.isSignlessInteger(1) ? 8 : integerType.getWidth();
      dtypeCode = integerType.isSignlessInteger(1)
                      ? kDLBool
                      : (integerType.isUnsignedInteger() ? kDLUInt : kDLInt);
    } else {
      return op.emitError("unsupported literal element type");
    }

    mlir::Value const dtypeArgument =
        tvm_ffi::DTypeType::build(rewriter, loc, dtypeCode, elementBits);
    mlir::FailureOr<mlir::Value> dtypeResult =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.runtime.tvm_ffi_to_torch_type",
            {dtypeArgument});
    if (mlir::failed(dtypeResult)) {
      return op.emitError("failed to call TVM FFI dtype conversion helper");
    }
    mlir::Value const dtypePayload =
        tvm_ffi::TVMFFIABIType::load(rewriter, loc, dtypeResult.value());
    mlir::Value const dtype =
        mlir::LLVM::TruncOp::create(rewriter, loc, i32Type, dtypePayload);
    llvm::ArrayRef<int64_t> const shape = denseType.getShape();
    mlir::Value const rank = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64Type, static_cast<int64_t>(shape.size()));
    mlir::Value const shapeCount = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64Type, static_cast<int64_t>(shape.size()));
    mlir::Value const sizes = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrType, i64Type, shapeCount);
    mlir::Value const strides = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrType, i64Type, shapeCount);
    int64_t stride = 1;
    for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0;
         --index) {
      mlir::Value const sizeSlot = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, i64Type, sizes,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::Value const strideSlot = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, i64Type, strides,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::LLVM::StoreOp::create(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, shape[index]),
          sizeSlot);
      mlir::LLVM::StoreOp::create(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, stride),
          strideSlot);
      stride *= shape[index];
    }

    mlir::Type storageType = denseType.getElementType();
    if (mlir::IntegerType const integerType =
            mlir::dyn_cast<mlir::IntegerType>(storageType);
        integerType && integerType.isSignlessInteger(1)) {
      storageType = i8Type;
    }
    mlir::Value const elementCount = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64Type, dense.getNumElements());
    mlir::Value data = mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                                    storageType, elementCount);
    auto storeElement = [&](int64_t index, mlir::Attribute value) {
      mlir::Value const slot = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, storageType, data,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::LLVM::StoreOp::create(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, storageType, value),
          slot);
    };
    if (mlir::isa<mlir::FloatType>(storageType)) {
      for (auto [index, value] :
           llvm::enumerate(dense.getValues<llvm::APFloat>())) {
        storeElement(static_cast<int64_t>(index),
                     mlir::FloatAttr::get(storageType, value));
      }
    } else {
      for (auto [index, value] :
           llvm::enumerate(dense.getValues<llvm::APInt>())) {
        if (mlir::cast<mlir::IntegerType>(denseType.getElementType())
                .isSignlessInteger(1)) {
          value = llvm::APInt(8, value.getBoolValue());
        }
        storeElement(static_cast<int64_t>(index),
                     mlir::IntegerAttr::get(storageType, value));
      }
    }

    auto convertDevice =
        [&](int32_t deviceType) -> mlir::FailureOr<mlir::Value> {
      mlir::Value const argument = tvm_ffi::IntType::build(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Type, deviceType));
      mlir::FailureOr<mlir::Value> result =
          conversion::utils::callTVMFFIGlobalFunction(
              rewriter, loc, module,
              "trident.runtime.tvm_ffi_device_to_torch_device_type",
              {argument});
      if (mlir::failed(result)) {
        return mlir::failure();
      }
      mlir::Value const deviceTypePayload =
          tvm_ffi::TVMFFIABIType::load(rewriter, loc, result.value());
      return mlir::LLVM::TruncOp::create(rewriter, loc, i32Type,
                                         deviceTypePayload)
          .getResult();
    };
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, 1);
    mlir::Value const zeroI32 =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Type, 0);
    mlir::Value const zeroI64 =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, 0);
    mlir::Value const deviceIndexSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, i32Type, one);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(
                module)),
        deviceIndexSlot);
    mlir::Value const deviceIndex =
        mlir::LLVM::LoadOp::create(rewriter, loc, i32Type, deviceIndexSlot);
    mlir::Value const cpuOutput =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, one);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchCreateTensorFromBlob(
                module)),
        mlir::ValueRange{data, rank, sizes, strides, zeroI64, dtype,
                         TRIDENT_CHECK_FAILURE(convertDevice(kDLCPU)), zeroI32,
                         cpuOutput});
    mlir::Value const cpuTensor =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrType, cpuOutput);
    mlir::Value const cudaOutput =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, one);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchEmptyStrided(module)),
        mlir::ValueRange{rank, sizes, strides, dtype,
                         TRIDENT_CHECK_FAILURE(convertDevice(kDLCUDA)),
                         deviceIndex, cudaOutput});
    mlir::Value const tensor =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrType, cudaOutput);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchCopy_(module)),
        mlir::ValueRange{tensor, cpuTensor, zeroI32});
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchDeleteTensorObject(module)),
        cpuTensor);

    mlir::Value const tensorArgument =
        tvm_ffi::IntType::build(rewriter, loc, 0);
    mlir::Value const opaquePtrTypeIndex = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i32Type, kTVMFFIOpaquePtr);
    mlir::LLVM::StoreOp::create(
        rewriter, loc, opaquePtrTypeIndex,
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrType,
                                  tvm_ffi::TVMFFICABIType::getLLVMType(context),
                                  tensorArgument,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
    mlir::Value const opaquePtrPayload =
        mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Type, tensor);
    mlir::LLVM::StoreOp::create(
        rewriter, loc, opaquePtrPayload,
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrType,
                                  tvm_ffi::TVMFFICABIType::getLLVMType(context),
                                  tensorArgument,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
    mlir::FailureOr<mlir::Value> result =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.runtime.tensor_to_tvm_ffi_object",
            {tensorArgument});
    if (mlir::failed(result)) {
      return op.emitError("failed to call TVM FFI tensor conversion helper");
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
        op, tvm_ffi::TVMFFICABIType::getLLVMType(context), result.value());
    return mlir::success();
  }
};

class ConvertToOp final : public mlir::OpConversionPattern<tvm_ffi::ToOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ToOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::Value value = adaptor.getValue();
    mlir::Type const resultType = op.getResult().getType();
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    int32_t typeIndex;
    if (mlir::isa<tvm_ffi::BoolType>(resultType)) {
      value = mlir::LLVM::ZExtOp::create(rewriter, loc, i64Ty, value);
      typeIndex = tvm_ffi::BoolType::getTypeIndex();
    } else if (mlir::isa<tvm_ffi::FloatType>(resultType)) {
      value = mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty, value);
      typeIndex = tvm_ffi::FloatType::getTypeIndex();
    } else if (mlir::isa<tvm_ffi::IntType>(resultType) &&
               value.getType().isSignlessInteger(64)) {
      typeIndex = tvm_ffi::IntType::getTypeIndex();
    } else {
      return mlir::failure();
    }
    rewriter.replaceOp(op, createConstantABI(rewriter, loc, typeIndex, value));
    return mlir::success();
  }
};

struct TVMFFIToLLVMDialectInterface final
    : public mlir::ConvertToLLVMPatternInterface {
  explicit TVMFFIToLLVMDialectInterface(mlir::Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  typeConverter.addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    return llvm::TypeSwitch<mlir::Type, std::optional<mlir::Type>>(type)
        .Case<tvm_ffi::PackArgumentsType>(
            [](tvm_ffi::PackArgumentsType type) -> std::optional<mlir::Type> {
              return mlir::LLVM::LLVMStructType::getLiteral(
                  type.getContext(),
                  {mlir::LLVM::LLVMPointerType::get(type.getContext()),
                   mlir::IntegerType::get(type.getContext(), 32)});
            })
        .Case<dlpack::DLTensorType>(
            [](dlpack::DLTensorType type) -> std::optional<mlir::Type> {
              return dlpack::DLTensorType::getLLVMType(type.getContext());
            })
        .Case<tvm_ffi::ArrayType, tvm_ffi::ExceptionType, tvm_ffi::FunctionType,
              tvm_ffi::StrType, tvm_ffi::TensorType, tvm_ffi::TupleType>(
            [](mlir::Type) -> std::optional<mlir::Type> {
              return mlir::Type();
            })
        .Case<tvm_ffi::CFunctionType, tvm_ffi::ObjectType>(
            [](mlir::Type type) -> std::optional<mlir::Type> {
              return mlir::LLVM::LLVMPointerType::get(type.getContext());
            })
        .Case<tvm_ffi::TVMFFICOpaqueABIType>(
            [](tvm_ffi::TVMFFICOpaqueABIType) -> std::optional<mlir::Type> {
              return std::nullopt;
            })
        .Case<tvm_ffi::TVMFFICABIType>(
            [](tvm_ffi::TVMFFICABIType type) -> std::optional<mlir::Type> {
              return tvm_ffi::TVMFFICABIType::getLLVMType(type.getContext());
            })
        .Default([](mlir::Type) -> std::optional<mlir::Type> {
          return std::nullopt;
        });
  });
  mlir::registerConversionPDLFunctions(patterns);
  populateGeneratedPDLLPatterns(patterns,
                                mlir::PDLConversionConfig(&typeConverter));
  patterns
      .add<ConvertAsOp, ConvertConstantDTypeOp, ConvertConstantDeviceOp,
           ConvertConstantRawStrOp, ConvertEqOp, ConvertExceptionOp,
           ConvertFunctionCallOp, ConvertFunctionGetGlobalOp, ConvertGetOp,
           ConvertIsaOp, ConvertLocalCallOp<tvm_ffi::CallOp>,
           ConvertLocalCallOp<tvm_ffi::PackCallOp>, ConvertPackGetOp,
           ConvertPackSizeOp,
           ConvertRefOp<tvm_ffi::ObjectDecRefOp,
                        &conversion::utils::getOrCreateTVMFFIObjectDecRef>,
           ConvertRefOp<tvm_ffi::ObjectIncRefOp,
                        &conversion::utils::getOrCreateTVMFFIObjectIncRef>,
           ConvertTVMFFIDowncastOp, ConvertTVMFFIUpcastOp, ConvertTensorCloneOp,
           ConvertTensorCopyOp, ConvertTensorLiteralOp, ConvertToOp>(
          typeConverter, patterns.getContext());
  target.addIllegalDialect<tvm_ffi::TVMFFIDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *, tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace trident::conversion
