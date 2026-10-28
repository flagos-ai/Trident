//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToAscend/TVMFFIToAscend.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/CFunctionDeclUtils.h"
#include "trident/core/Conversion/Utils/KernelLaunchConversion.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Runtime/AscendRuntime.h"
#include <algorithm>
#include <dlpack/dlpack.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Transforms/DialectConversion.h>
#include <cstdint>
#include <string>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOASCEND
#include "trident/core/Conversion/Passes.h.inc"

static mlir::FailureOr<mlir::Value>
toStorageBits(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              mlir::Value value) {
  mlir::Type const i64 = rewriter.getI64Type();
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
    return mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64, value).getResult();
  }
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
    if (integer.getWidth() < 64) {
      return mlir::arith::ExtUIOp::create(rewriter, loc, i64, value).getResult();
    }
    if (integer.getWidth() > 64) {
      return mlir::arith::TruncIOp::create(rewriter, loc, i64, value).getResult();
    }
    return value;
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(value.getType())) {
    mlir::IntegerType const bitsType =
        rewriter.getIntegerType(floating.getWidth());
    mlir::Value bits =
        mlir::arith::BitcastOp::create(rewriter, loc, bitsType, value);
    if (floating.getWidth() < 64) {
      bits = mlir::arith::ExtUIOp::create(rewriter, loc, i64, bits);
    }
    return bits;
  }
  return mlir::failure();
}

static mlir::LLVM::GlobalOp
createBytesGlobal(mlir::ConversionPatternRewriter &rewriter,
                  mlir::ModuleOp module, mlir::Location loc,
                  llvm::StringRef name, llvm::StringRef bytes,
                  bool nullTerminate = false) {
  if (auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
    return global;
  }
  std::string storage = bytes.str();
  if (nullTerminate) {
    storage.push_back('\0');
  }
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto type = mlir::LLVM::LLVMArrayType::get(rewriter.getI8Type(),
                                              storage.size());
  return mlir::LLVM::GlobalOp::create(
      rewriter, loc, type, /*isConstant=*/true,
      mlir::LLVM::Linkage::Internal, name, rewriter.getStringAttr(storage),
      /*alignment=*/0);
}

static mlir::FailureOr<uint8_t> getArgumentCode(llvm::StringRef type) {
  if (!type.empty() && type.front() == '*') {
    return TRIDENT_ASCEND_POINTER;
  }
  if (type == "i1" || type == "u1" || type == "i32" || type == "u32") {
    return TRIDENT_ASCEND_I32;
  }
  if (type == "i8" || type == "u8") {
    return TRIDENT_ASCEND_I8;
  }
  if (type == "i16" || type == "u16") {
    return TRIDENT_ASCEND_I16;
  }
  if (type == "i64" || type == "u64") {
    return TRIDENT_ASCEND_I64;
  }
  if (type == "fp16" || type == "bf16" || type == "fp32" || type == "f32") {
    return TRIDENT_ASCEND_F32;
  }
  if (type == "fp64") {
    return TRIDENT_ASCEND_F64;
  }
  return mlir::failure();
}

class ConvertKernelLaunchOp final
    : public mlir::OpConversionPattern<tvm_ffi::KernelLaunchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::KernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitOpError("op is not inside a ModuleOp");
    }
    auto binary = module.lookupSymbol<tvm_ffi::AscendKernelOp>(
        op.getKernel().getRootReference());
    if (!binary) {
      return op.emitOpError("does not reference a tvm_ffi.ascend_kernel");
    }
    mlir::ArrayAttr const specializations = op.getSpecializationsAttr();
    if (mlir::failed(utils::buildKernelSpecializationGuard(
            op, *getTypeConverter(), rewriter))) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> values;
    for (auto [specialization, source] : llvm::zip(
             specializations.getAsRange<tvm_ffi::SpecializationAttrInterface>(),
             op.getKernelOperands())) {
      if (mlir::isa<tvm_ffi::ConstantSpecializationAttr>(specialization)) {
        continue;
      }
      mlir::Value operand = utils::materializeKernelOperand(
          *getTypeConverter(), rewriter, loc, specialization, source);
      if (!operand) {
        return op.emitOpError("failed to materialize a native kernel operand");
      }
      mlir::FailureOr<mlir::Value> bits = toStorageBits(rewriter, loc, operand);
      if (mlir::failed(bits)) {
        return op.emitOpError("unsupported native Ascend argument type ")
               << operand.getType();
      }
      values.push_back(*bits);
    }
    if (values.size() != binary.getArgumentTypes().size()) {
      return op.emitOpError("runtime argument count does not match binary ABI");
    }

    llvm::SmallVector<char> typeCodes;
    for (mlir::Attribute attribute : binary.getArgumentTypes()) {
      auto typeName = mlir::cast<mlir::StringAttr>(attribute);
      mlir::FailureOr<uint8_t> code = getArgumentCode(typeName.getValue());
      if (mlir::failed(code)) {
        return binary.emitOpError("unsupported Ascend ABI type ")
               << typeName.getValue();
      }
      typeCodes.push_back(static_cast<char>(*code));
    }

    auto denseBinary = mlir::cast<mlir::DenseElementsAttr>(binary.getBinary());
    llvm::ArrayRef<char> const rawBinary = denseBinary.getRawData();
    llvm::StringRef binaryBytes(rawBinary.data(), rawBinary.size());
    std::string const prefix = binary.getSymName().str();
    mlir::LLVM::GlobalOp binaryGlobal = createBytesGlobal(
        rewriter, module, loc, prefix + "_npubin", binaryBytes);
    mlir::LLVM::GlobalOp nameGlobal =
        createBytesGlobal(rewriter, module, loc, prefix + "_name",
                          binary.getKernelName(), /*nullTerminate=*/true);
    mlir::LLVM::GlobalOp modeGlobal =
        createBytesGlobal(rewriter, module, loc, prefix + "_mode",
                          binary.getMixMode(), /*nullTerminate=*/true);
    if (typeCodes.empty()) {
      typeCodes.push_back(0);
    }
    mlir::LLVM::GlobalOp typesGlobal = createBytesGlobal(
        rewriter, module, loc, prefix + "_types",
        llvm::StringRef(typeCodes.data(), typeCodes.size()));

    mlir::Type const i64 = rewriter.getI64Type();
    mlir::Value valueCount = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64, std::max<size_t>(1, values.size()));
    mlir::Value valuesSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        i64, valueCount);
    for (auto [index, value] : llvm::enumerate(values)) {
      mlir::Value offset = mlir::LLVM::ConstantOp::create(
          rewriter, loc, i64, static_cast<int64_t>(index));
      mlir::Value slot = mlir::LLVM::GEPOp::create(
          rewriter, loc,
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext()), i64,
          valuesSlot, mlir::ValueRange{offset});
      mlir::LLVM::StoreOp::create(rewriter, loc, value, slot);
    }

    auto launch = utils::getOrCreateCAPI<decltype(&tridentAscendLaunch)>(
        module, "tridentAscendLaunch");
    if (mlir::failed(launch)) {
      return op.emitOpError("failed to declare Ascend runtime launch API");
    }
    auto getDeviceIndex =
        utils::getOrCreateAOTITorchGetCurrentDeviceIndex(module);
    auto getStream = utils::getOrCreateTVMFFIEnvGetStream(module);
    if (mlir::failed(getDeviceIndex) || mlir::failed(getStream)) {
      return op.emitOpError("failed to declare Ascend device APIs");
    }
    mlir::Value deviceSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        rewriter.getI32Type(),
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64, 1));
    mlir::LLVM::CallOp::create(rewriter, loc, *getDeviceIndex, deviceSlot);
    mlir::Value deviceIndex = mlir::LLVM::LoadOp::create(
        rewriter, loc, rewriter.getI32Type(), deviceSlot);
    mlir::Value deviceType = mlir::LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI32Type(), DLDeviceType::kDLExtDev);
    mlir::Value stream =
        mlir::LLVM::CallOp::create(
            rewriter, loc, *getStream,
            mlir::ValueRange{deviceType, deviceIndex})
            .getResult();
    auto address = [&](mlir::LLVM::GlobalOp global) -> mlir::Value {
      return mlir::LLVM::AddressOfOp::create(rewriter, loc, global);
    };
    auto constantI64 = [&](int64_t value) -> mlir::Value {
      return mlir::LLVM::ConstantOp::create(rewriter, loc, i64, value);
    };
    auto constantI8 = [&](bool value) -> mlir::Value {
      return mlir::LLVM::ConstantOp::create(rewriter, loc,
                                            rewriter.getI8Type(), value);
    };
    llvm::SmallVector<mlir::Value> arguments{
        address(binaryGlobal),
        constantI64(binaryBytes.size()),
        address(nameGlobal),
        address(modeGlobal),
        address(typesGlobal),
        constantI64(values.size()),
        valuesSlot,
        deviceIndex,
        stream,
        adaptor.getGridSizeX(),
        adaptor.getGridSizeY(),
        adaptor.getGridSizeZ(),
        constantI64(binary.getWorkspaceSize()),
        constantI64(binary.getLockNum()),
        constantI64(binary.getLockInitValue()),
        constantI8(binary.getHasFfts()),
        constantI8(binary.getForceSimtOnly()),
        constantI8(binary.getUseV2Launch()),
        constantI8(binary.getEnableTaskQueue()),
        constantI8(binary.getAutoMapBlocks()),
        constantI64(binary.getSharedMemorySize()),
        constantI64(binary.getPhysicalBlockCount())};
    mlir::Value status =
        mlir::LLVM::CallOp::create(rewriter, loc, *launch, arguments)
            .getResult();
    mlir::Value ok = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, status,
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), 0));
    mlir::cf::AssertOp::create(rewriter, loc, ok,
                               "Ascend kernel launch failed");
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertTVMFFIToAscendPass final
    : public impl::ConvertTVMFFIToAscendBase<ConvertTVMFFIToAscendPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::TypeConverter typeConverter;
    utils::configureKernelLaunchTypeConverter(typeConverter);

    target.addIllegalOp<tvm_ffi::KernelLaunchOp>();
    target.addLegalDialect<dlpack::DLPackDialect, mlir::BuiltinDialect,
                           mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::func::FuncDialect,
                           tvm_ffi::TVMFFIDialect>();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertKernelLaunchOp>(typeConverter, &getContext());
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    llvm::SmallVector<tvm_ffi::AscendKernelOp> binaries;
    getOperation().walk(
        [&](tvm_ffi::AscendKernelOp op) { binaries.push_back(op); });
    for (tvm_ffi::AscendKernelOp binary : binaries) {
      binary.erase();
    }
  }
};

} // namespace trident::conversion
