//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <string>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

namespace {

llvm::SmallVector<int32_t> getExpectedTypeIndices(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, llvm::SmallVector<int32_t>>(type)
      .Case<tvm_ffi::UnionType>(
          [](tvm_ffi::UnionType unionType) -> llvm::SmallVector<int32_t> {
            return llvm::map_to_vector(
                unionType.getTypes(), [](mlir::Type member) -> int32_t {
                  return mlir::cast<tvm_ffi::TVMFFITypeIndexInterface>(member)
                      .getTypeIndex();
                });
          })
      .Case<tvm_ffi::AnyType>(
          [](tvm_ffi::AnyType) -> llvm::SmallVector<int32_t> { return {}; })
      .Case<tvm_ffi::TVMFFIABIType>(
          [](tvm_ffi::TVMFFIABIType abiType) -> llvm::SmallVector<int32_t> {
            return {mlir::cast<tvm_ffi::TVMFFITypeIndexInterface>(abiType)
                        .getTypeIndex()};
          })
      .Default([](mlir::Type) -> llvm::SmallVector<int32_t> { return {}; });
}

mlir::LogicalResult lowerReturn(tvm_ffi::ReturnOp op, mlir::OpBuilder &builder,
                                mlir::ValueRange slots) {
  builder.setInsertionPoint(op);
  mlir::Value ready =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), true, 1);
  for (mlir::Value const value : op.getOperands()) {
    tvm_ffi::ResultType const type =
        mlir::cast<tvm_ffi::ResultType>(value.getType());
    mlir::UnrealizedConversionCastOp fields =
        value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!fields || fields.getNumOperands() != 2 ||
        !fields.getInputs().back().getType().isInteger(1) ||
        fields.getInputs().front().getType() != type.getPayloadType()) {
      return op.emitError("run lower-tvm-ffi-monad before converting "
                          "tvm_ffi.return to the ABI");
    }
    ready = builder.createOrFold<mlir::arith::AndIOp>(
        op.getLoc(), ready, fields.getInputs().back());
  }
  llvm::SmallVector<mlir::Value> const payloads = llvm::map_to_vector(
      op.getOperands(), [](mlir::Value value) -> mlir::Value {
        return value.getDefiningOp<mlir::UnrealizedConversionCastOp>()
            .getInputs()
            .front();
      });
  mlir::LLVMTypeConverter const converter(builder.getContext());
  llvm::SmallVector<mlir::Type> const storageTypes =
      llvm::map_to_vector(payloads, [&](mlir::Value payload) -> mlir::Type {
        return mlir::isa<tvm_ffi::TVMFFIABIType>(payload.getType())
                   ? tvm_ffi::TVMFFIABIType::getLLVMType(builder.getContext())
                   : converter.convertType(payload.getType());
      });
  if (llvm::any_of(storageTypes,
                   [](mlir::Type type) -> bool { return !type; })) {
    return op.emitError(
        "payload type cannot be lowered to an LLVM output slot");
  }
  mlir::scf::IfOp branch = mlir::scf::IfOp::create(
      builder, op.getLoc(), mlir::TypeRange{builder.getI32Type()}, ready, true);
  builder.setInsertionPointToStart(branch.thenBlock());
  for (auto [payload, slot, type] :
       llvm::zip_equal(payloads, slots, storageTypes)) {
    mlir::Value value = payload;
    if (value.getType() != type) {
      value = mlir::UnrealizedConversionCastOp::create(builder, op.getLoc(),
                                                       type, value)
                  .getResult(0);
    }
    mlir::LLVM::StoreOp::create(builder, op.getLoc(), value, slot);
  }
  mlir::Value const zero =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 32);
  mlir::scf::YieldOp::create(builder, op.getLoc(), zero);
  builder.setInsertionPointToStart(branch.elseBlock());
  mlir::Value const error =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), -1, 32);
  mlir::scf::YieldOp::create(builder, op.getLoc(), error);
  builder.setInsertionPoint(op);
  mlir::func::ReturnOp::create(builder, op.getLoc(), branch.getResults());
  op.erase();
  return mlir::success();
}

} // namespace

class ConvertTVMFFIToFuncPass final
    : public impl::ConvertTVMFFIToFuncBase<ConvertTVMFFIToFuncPass> {
  void runOnOperation() final {
    getOperation().walk([&](tvm_ffi::FuncOp tvmffiFuncOp) -> mlir::WalkResult {
      mlir::OpBuilder builder(tvmffiFuncOp);
      const mlir::FunctionType targetType = tvmffiFuncOp.getFunctionType();
      llvm::SmallVector<mlir::Type> const payloadTypes = llvm::map_to_vector(
          targetType.getResults(), [](mlir::Type result) -> mlir::Type {
            return mlir::cast<tvm_ffi::ResultType>(result).getPayloadType();
          });
      llvm::SmallVector<mlir::Type> argumentsTypes(targetType.getInputs());
      argumentsTypes.append(
          payloadTypes.size(),
          mlir::LLVM::LLVMPointerType::get(builder.getContext()));
      mlir::FunctionType const implementationType = builder.getFunctionType(
          argumentsTypes, mlir::TypeRange{builder.getI32Type()});
      mlir::func::FuncOp funcOp = mlir::func::FuncOp::create(
          builder, tvmffiFuncOp.getLoc(), tvmffiFuncOp.getSymName(),
          implementationType);
      if (tvmffiFuncOp.getSymVisibility()) {
        funcOp.setSymVisibilityAttr(tvmffiFuncOp.getSymVisibilityAttr());
      }
      funcOp.getBody().takeBody(tvmffiFuncOp.getBody());
      if (funcOp.isExternal()) {
        funcOp.setPrivate();
        tvmffiFuncOp.erase();
        return mlir::WalkResult::advance();
      }
      llvm::SmallVector<mlir::Value> const slots =
          llvm::map_to_vector(payloadTypes, [&](mlir::Type) -> mlir::Value {
            return funcOp.getBody().front().addArgument(
                mlir::LLVM::LLVMPointerType::get(builder.getContext()),
                tvmffiFuncOp.getLoc());
          });
      const mlir::WalkResult returnConversionResult =
          funcOp.walk([&](tvm_ffi::ReturnOp returned) -> mlir::WalkResult {
            if (mlir::failed(lowerReturn(returned, builder, slots))) {
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          });
      if (returnConversionResult.wasInterrupted()) {
        signalPassFailure();
        return mlir::WalkResult::interrupt();
      }
      if (tvmffiFuncOp.getEmitTvmFfiAbi()) {
        mlir::MLIRContext *context = builder.getContext();
        mlir::ModuleOp module = tvmffiFuncOp->getParentOfType<mlir::ModuleOp>();
        if (!module) {
          tvmffiFuncOp.emitError("expected TVM FFI function inside a module");
          signalPassFailure();
          return mlir::WalkResult::interrupt();
        }
        const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
        const mlir::FunctionType wrapperType = mlir::FunctionType::get(
            context, {ptrTy, ptrTy, builder.getI32Type(), ptrTy},
            {builder.getI32Type()});
        const std::string wrapperName =
            llvm::formatv("__tvm_ffi_{0}", tvmffiFuncOp.getSymName()).str();
        builder.setInsertionPointAfter(funcOp);
        mlir::func::FuncOp wrapper = mlir::func::FuncOp::create(
            builder, tvmffiFuncOp.getLoc(), wrapperName, wrapperType);
        mlir::Block *entry = wrapper.addEntryBlock();
        const mlir::Value argsPtr = entry->getArgument(1);
        const mlir::Value resultPtr = entry->getArgument(3);
        const mlir::IntegerType i32Ty = builder.getI32Type();
        const mlir::IntegerType i64Ty = builder.getI64Type();
        mlir::LLVM::LLVMStructType const llvmAnyTy =
            tvm_ffi::TVMFFIABIType::getLLVMType(context);
        mlir::LLVM::LLVMPointerType const llvmPtrTy =
            mlir::LLVM::LLVMPointerType::get(context);

        builder.setInsertionPointToStart(entry);
        llvm::SmallVector<mlir::Value> arguments;
        mlir::Value allTypesMatch = mlir::arith::ConstantOp::create(
            builder, tvmffiFuncOp.getLoc(), builder.getI1Type(),
            builder.getBoolAttr(true));
        for (auto [index, type] :
             llvm::enumerate(tvmffiFuncOp.getArgumentTypes())) {
          llvm::SmallVector<int32_t> const expected =
              getExpectedTypeIndices(type);
          if (expected.empty()) {
            signalPassFailure();
            return mlir::WalkResult::interrupt();
          }
          const mlir::Value slot = mlir::LLVM::GEPOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmPtrTy, llvmAnyTy, argsPtr,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
          const mlir::Value value = mlir::LLVM::LoadOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmAnyTy, slot);
          arguments.push_back(mlir::UnrealizedConversionCastOp::create(
                                  builder, tvmffiFuncOp.getLoc(), type, value)
                                  .getResult(0));
          const mlir::Value actual = mlir::LLVM::ExtractValueOp::create(
              builder, tvmffiFuncOp.getLoc(), i32Ty, value,
              llvm::ArrayRef<int64_t>{0});
          mlir::Value condition;
          for (int32_t const expectedIndex : expected) {
            mlir::Value const matches = mlir::LLVM::ICmpOp::create(
                builder, tvmffiFuncOp.getLoc(), mlir::LLVM::ICmpPredicate::eq,
                actual,
                mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                               i32Ty, expectedIndex));
            condition =
                condition
                    ? mlir::arith::OrIOp::create(builder, tvmffiFuncOp.getLoc(),
                                                 condition, matches)
                    : matches;
          }
          allTypesMatch = mlir::arith::AndIOp::create(
              builder, tvmffiFuncOp.getLoc(), allTypesMatch, condition);
        }
        mlir::scf::IfOp guard = mlir::scf::IfOp::create(
            builder, tvmffiFuncOp.getLoc(),
            mlir::TypeRange{builder.getI32Type()}, allTypesMatch,
            /*withElseRegion=*/true);
        builder.setInsertionPointToStart(guard.thenBlock());
        if (!llvm::all_of(payloadTypes, [](mlir::Type type) -> bool {
              return mlir::isa<tvm_ffi::TVMFFIABIType>(type);
            })) {
          tvmffiFuncOp.emitError(
              "ABI return payloads must have a TVM FFI ABI type");
          signalPassFailure();
          return mlir::WalkResult::interrupt();
        }
        llvm::SmallVector<mlir::Value> const slots =
            llvm::map_to_vector(payloadTypes, [&](mlir::Type) -> mlir::Value {
              return mlir::LLVM::AllocaOp::create(
                  builder, tvmffiFuncOp.getLoc(), llvmPtrTy, llvmAnyTy,
                  mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                                 i64Ty, 1));
            });
        llvm::append_range(arguments, slots);
        mlir::func::CallOp call = mlir::func::CallOp::create(
            builder, tvmffiFuncOp.getLoc(), funcOp.getSymName(),
            funcOp.getFunctionType().getResults(), arguments);
        mlir::Value const zero = mlir::arith::ConstantIntOp::create(
            builder, tvmffiFuncOp.getLoc(), 0, 32);
        mlir::Value const success = mlir::arith::CmpIOp::create(
            builder, tvmffiFuncOp.getLoc(), mlir::arith::CmpIPredicate::eq,
            call.getResult(0), zero);
        mlir::scf::IfOp successful =
            mlir::scf::IfOp::create(builder, tvmffiFuncOp.getLoc(),
                                    mlir::TypeRange{i32Ty}, success, true);
        builder.setInsertionPointToStart(successful.thenBlock());
        // Every Result has one ABI payload, including None. Read output slots
        // only after success, and reuse those slots when packing multiple
        // values.
        mlir::Value resultSlot = slots.front();
        if (slots.size() > 1) {
          mlir::FailureOr<mlir::Value> array = utils::callTVMFFIGlobalFunction(
              builder, tvmffiFuncOp.getLoc(), module, "ffi.Array", slots);
          if (mlir::failed(array)) {
            tvmffiFuncOp.emitError("failed to create ffi.Array result");
            signalPassFailure();
            return mlir::WalkResult::interrupt();
          }
          resultSlot = *array;
        }
        mlir::Value const packed = mlir::LLVM::LoadOp::create(
            builder, tvmffiFuncOp.getLoc(), llvmAnyTy, resultSlot);
        mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(), packed,
                                    resultPtr);
        mlir::Value const packedSuccess = mlir::LLVM::ConstantOp::create(
            builder, tvmffiFuncOp.getLoc(), i32Ty, 0);
        mlir::scf::YieldOp::create(builder, tvmffiFuncOp.getLoc(),
                                   packedSuccess);
        builder.setInsertionPointToStart(successful.elseBlock());
        mlir::scf::YieldOp::create(builder, tvmffiFuncOp.getLoc(),
                                   call.getResult(0));
        builder.setInsertionPointAfter(successful);
        mlir::scf::YieldOp::create(builder, tvmffiFuncOp.getLoc(),
                                   successful.getResults());
        builder.setInsertionPointToStart(guard.elseBlock());
        const mlir::Value kindPtr =
            utils::getString(builder, tvmffiFuncOp.getLoc(), "GuardMatch");
        mlir::Value exceptionValue = mlir::LLVM::UndefOp::create(
            builder, tvmffiFuncOp.getLoc(), llvmAnyTy);
        exceptionValue = mlir::LLVM::InsertValueOp::create(
            builder, tvmffiFuncOp.getLoc(), exceptionValue,
            mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i32Ty, 8),
            llvm::ArrayRef<int64_t>{0});
        exceptionValue = mlir::LLVM::InsertValueOp::create(
            builder, tvmffiFuncOp.getLoc(), exceptionValue,
            mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i32Ty, 0),
            llvm::ArrayRef<int64_t>{1});
        exceptionValue = mlir::LLVM::InsertValueOp::create(
            builder, tvmffiFuncOp.getLoc(), exceptionValue,
            mlir::LLVM::PtrToIntOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i64Ty, kindPtr),
            llvm::ArrayRef<int64_t>{2});
        const mlir::Value exceptionSlot = mlir::LLVM::AllocaOp::create(
            builder, tvmffiFuncOp.getLoc(), ptrTy, llvmAnyTy,
            mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                           i64Ty, 1));
        mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(),
                                    exceptionValue, exceptionSlot);
        mlir::FailureOr<mlir::Value> exceptionResult =
            conversion::utils::callTVMFFIGlobalFunction(
                builder, tvmffiFuncOp.getLoc(), module, "trident.ffi.Exception",
                llvm::ArrayRef<mlir::Value>{exceptionSlot});
        if (mlir::failed(exceptionResult)) {
          tvmffiFuncOp.emitError("failed to create guard exception");
          signalPassFailure();
          return mlir::WalkResult::interrupt();
        }
        const mlir::Value exception =
            mlir::LLVM::LoadOp::create(builder, tvmffiFuncOp.getLoc(),
                                       llvmAnyTy, exceptionResult.value())
                .getResult();
        mlir::LLVM::StoreOp::create(builder, tvmffiFuncOp.getLoc(), exception,
                                    resultPtr);
        mlir::Value const guardSuccess = mlir::LLVM::ConstantOp::create(
            builder, tvmffiFuncOp.getLoc(), i32Ty, 0);
        mlir::scf::YieldOp::create(builder, tvmffiFuncOp.getLoc(),
                                   guardSuccess);
        builder.setInsertionPointAfter(guard);
        mlir::func::ReturnOp::create(builder, tvmffiFuncOp.getLoc(),
                                     guard.getResults());
      }
      tvmffiFuncOp->erase();
      return mlir::WalkResult::advance();
    });
  }
};

} // namespace trident::conversion
