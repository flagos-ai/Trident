//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
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

static mlir::LogicalResult lowerReturn(tvm_ffi::ReturnOp op,
                                       mlir::OpBuilder &builder,
                                       mlir::ValueRange slots,
                                       mlir::Type payloadType) {
  builder.setInsertionPoint(op);
  mlir::Value const ready = tvm_ffi::GetOp::create(
      builder, op.getLoc(), builder.getI1Type(), op.getSuccess());
  mlir::scf::IfOp branch = mlir::scf::IfOp::create(
      builder, op.getLoc(), mlir::TypeRange{builder.getI32Type()}, ready, true);
  builder.setInsertionPointToStart(branch.thenBlock());
  mlir::Type const anyType =
      tvm_ffi::TVMFFICABIType::getLLVMType(builder.getContext());
  mlir::Value const checked = builder.createOrFold<tvm_ffi::DowncastOp>(
      op.getLoc(), payloadType, op.getValue());
  mlir::Value const payload = builder.createOrFold<tvm_ffi::UpcastOp>(
      op.getLoc(), tvm_ffi::CAnyType::get(builder.getContext()), checked);
  mlir::Value const value = mlir::UnrealizedConversionCastOp::create(
                                builder, op.getLoc(), anyType, payload)
                                .getResult(0);
  mlir::LLVM::StoreOp::create(builder, op.getLoc(), value, slots.front());
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

class ConvertTVMFFIToFuncPass final
    : public impl::ConvertTVMFFIToFuncBase<ConvertTVMFFIToFuncPass> {
  void runOnOperation() final {
    getOperation().walk([&](tvm_ffi::FuncOp tvmffiFuncOp) -> mlir::WalkResult {
      mlir::OpBuilder builder(tvmffiFuncOp);
      const mlir::FunctionType targetType = tvmffiFuncOp.getFunctionType();
      llvm::SmallVector<mlir::Type> const payloadTypes(targetType.getResults());
      llvm::SmallVector<mlir::Type> argumentsTypes(targetType.getInputs());
      argumentsTypes.append(
          payloadTypes.size(),
          mlir::LLVM::LLVMPointerType::get(builder.getContext()));
      mlir::FunctionType const implementationType = builder.getFunctionType(
          argumentsTypes, mlir::TypeRange{builder.getI32Type()});
      mlir::func::FuncOp funcOp = mlir::func::FuncOp::create(
          builder, tvmffiFuncOp.getLoc(), tvmffiFuncOp.getSymName(),
          implementationType);
      if (tvmffiFuncOp.getEmitTvmFfiAbi()) {
        funcOp->setAttr("emit_tvm_ffi_abi", builder.getUnitAttr());
      }
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
            if (mlir::failed(lowerReturn(returned, builder, slots,
                                         payloadTypes.front()))) {
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
            tvm_ffi::TVMFFICABIType::getLLVMType(context);
        mlir::LLVM::LLVMPointerType const llvmPtrTy =
            mlir::LLVM::LLVMPointerType::get(context);

        builder.setInsertionPointToStart(entry);
        llvm::SmallVector<mlir::Value> arguments;
        mlir::Value allTypesMatch = mlir::arith::ConstantOp::create(
            builder, tvmffiFuncOp.getLoc(), builder.getI1Type(),
            builder.getBoolAttr(true));
        for (auto [index, type] :
             llvm::enumerate(tvmffiFuncOp.getArgumentTypes())) {
          tvm_ffi::TVMFFITypeIndexInterface const indexed =
              mlir::dyn_cast<tvm_ffi::TVMFFITypeIndexInterface>(type);
          if (!indexed && !mlir::isa<tvm_ffi::CAnyType>(type)) {
            tvmffiFuncOp.emitError("unsupported packed argument type ") << type;
            signalPassFailure();
            return mlir::WalkResult::interrupt();
          }
          const mlir::Value slot = mlir::LLVM::GEPOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmPtrTy, llvmAnyTy, argsPtr,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
          const mlir::Value value = mlir::LLVM::LoadOp::create(
              builder, tvmffiFuncOp.getLoc(), llvmAnyTy, slot);
          mlir::Type const argumentType =
              mlir::isa<tvm_ffi::CFunctionType>(type)
                  ? mlir::Type(tvm_ffi::CAnyType::get(context))
                  : type;
          arguments.push_back(
              mlir::UnrealizedConversionCastOp::create(
                  builder, tvmffiFuncOp.getLoc(), argumentType, value)
                  .getResult(0));
          if (!indexed) {
            continue;
          }
          const mlir::Value actual = mlir::LLVM::ExtractValueOp::create(
              builder, tvmffiFuncOp.getLoc(), i32Ty, value,
              llvm::ArrayRef<int64_t>{0});
          mlir::Value const condition = mlir::LLVM::ICmpOp::create(
              builder, tvmffiFuncOp.getLoc(), mlir::LLVM::ICmpPredicate::eq,
              actual,
              mlir::LLVM::ConstantOp::create(builder, tvmffiFuncOp.getLoc(),
                                             i32Ty, indexed.getTypeIndex()));
          allTypesMatch = mlir::arith::AndIOp::create(
              builder, tvmffiFuncOp.getLoc(), allTypesMatch, condition);
        }
        mlir::scf::IfOp guard = mlir::scf::IfOp::create(
            builder, tvmffiFuncOp.getLoc(),
            mlir::TypeRange{builder.getI32Type()}, allTypesMatch,
            /*withElseRegion=*/true);
        builder.setInsertionPointToStart(guard.thenBlock());
        for (auto [index, type] :
             llvm::enumerate(tvmffiFuncOp.getArgumentTypes())) {
          if (mlir::isa<tvm_ffi::CFunctionType>(type)) {
            arguments[index] = tvm_ffi::DowncastOp::create(
                builder, tvmffiFuncOp.getLoc(), type, arguments[index]);
          }
        }
        if (!llvm::all_of(payloadTypes, [](mlir::Type type) -> bool {
              return mlir::isa<tvm_ffi::TVMFFICABIType>(type);
            })) {
          tvmffiFuncOp.emitError(
              "ABI return payloads must have a tagged TVM FFI ABI type");
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
        // The single Result payload, including None, is read only on success.
        mlir::Value const resultSlot = slots.front();
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
