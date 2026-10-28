//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Utils/KernelLaunchConversion.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace trident::conversion::utils {

void configureKernelLaunchTypeConverter(mlir::TypeConverter &typeConverter) {
  typeConverter.addConversion([](mlir::Type type) { return type; });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type resultType,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1) {
          return {};
        }
        mlir::Value input = inputs.front();
        if (mlir::isa<tvm_ffi::TensorType>(input.getType())) {
          if (!mlir::isa<mlir::LLVM::LLVMPointerType>(resultType)) {
            return {};
          }
          mlir::Value object = tvm_ffi::GetOp::create(
              builder, loc, tvm_ffi::ObjectType::get(builder.getContext()),
              input);
          mlir::Value tensor = tvm_ffi::AsOp::create(
              builder, loc, dlpack::DLTensorType::get(builder.getContext()),
              object);
          return dlpack::TensorDataOp::create(builder, loc, resultType, tensor);
        }
        if (mlir::isa<tvm_ffi::BoolType, tvm_ffi::FloatType,
                      tvm_ffi::IntType>(input.getType())) {
          input = tvm_ffi::GetOp::create(builder, loc, input);
        }
        mlir::Type const inputType = input.getType();
        if (inputType == resultType) {
          return input;
        }
        if (auto integer = mlir::dyn_cast<mlir::IntegerType>(inputType)) {
          auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(resultType);
          if (!targetInteger) {
            return {};
          }
          if (integer.getWidth() > targetInteger.getWidth()) {
            return mlir::arith::TruncIOp::create(builder, loc, resultType,
                                                 input);
          }
          if (integer.isInteger(1)) {
            return mlir::arith::ExtUIOp::create(builder, loc, resultType,
                                                input);
          }
          return mlir::arith::ExtSIOp::create(builder, loc, resultType, input);
        }
        if (mlir::isa<mlir::FloatType>(inputType) &&
            mlir::isa<mlir::FloatType>(resultType)) {
          return mlir::arith::TruncFOp::create(builder, loc, resultType, input);
        }
        return {};
      });
}

mlir::Value materializeKernelOperand(
    const mlir::TypeConverter &typeConverter,
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tvm_ffi::SpecializationAttrInterface specialization, mlir::Value source) {
  mlir::Type const kind = specialization.getTargetType();
  return source.getType() == kind
             ? source
             : typeConverter.materializeTargetConversion(rewriter, loc, kind,
                                                         source);
}

mlir::LogicalResult buildKernelSpecializationGuard(
    tvm_ffi::KernelLaunchOp op, const mlir::TypeConverter &typeConverter,
    mlir::ConversionPatternRewriter &rewriter) {
  tvm_ffi::FuncOp function = op->getParentOfType<tvm_ffi::FuncOp>();
  if (!function) {
    return mlir::success();
  }
  mlir::Location const loc = op.getLoc();
  mlir::Value allChecks = mlir::LLVM::ConstantOp::create(
      rewriter, loc, rewriter.getI1Type(), 1);
  for (auto [specialization, source] :
       llvm::zip(op.getSpecializationsAttr()
                     .getAsRange<tvm_ffi::SpecializationAttrInterface>(),
                 op.getKernelOperands())) {
    mlir::Value operand = materializeKernelOperand(
        typeConverter, rewriter, loc, specialization, source);
    if (!operand) {
      return op.emitOpError("failed to materialize a native kernel operand");
    }
    if (mlir::Value check =
            specialization.buildCheck(rewriter, loc, operand)) {
      allChecks = mlir::arith::AndIOp::create(rewriter, loc, allChecks, check);
    }
  }

  mlir::FunctionType const functionType = function.getFunctionType();
  if (functionType.getNumResults() != 1) {
    return op.emitOpError(
        "enclosing function must have one result to report "
        "specialization failure");
  }
  mlir::Type const resultType = functionType.getResult(0);
  bool const canReportFailure =
      llvm::TypeSwitch<mlir::Type, bool>(resultType)
          .Case<tvm_ffi::AnyType>([](tvm_ffi::AnyType) { return true; })
          .Case<tvm_ffi::UnionType>([&](tvm_ffi::UnionType type) {
            return type.contains(
                tvm_ffi::ExceptionType::get(function.getContext()));
          })
          .Default([](mlir::Type) { return false; });
  if (!canReportFailure) {
    return op.emitOpError(
        "enclosing function result must be !tvm_ffi.any or contain "
        "!tvm_ffi.exception");
  }

  mlir::Block *launchBlock = op->getBlock();
  mlir::Block *continuation =
      rewriter.splitBlock(launchBlock, op->getIterator());
  mlir::Block *failure =
      rewriter.createBlock(&function.getBody(), function.getBody().end());
  rewriter.setInsertionPointToEnd(launchBlock);
  mlir::cf::CondBranchOp::create(rewriter, loc, allChecks, continuation, {},
                                 failure, {});
  rewriter.setInsertionPointToEnd(failure);
  mlir::Value exception = tvm_ffi::ExceptionOp::create(
      rewriter, loc, tvm_ffi::ExceptionType::get(function.getContext()),
      "GuardMatch");
  mlir::Value failureResult =
      tvm_ffi::CastOp::create(rewriter, loc, resultType, exception);
  tvm_ffi::ReturnOp::create(rewriter, loc, failureResult);
  rewriter.setInsertionPoint(op);
  return mlir::success();
}

} // namespace trident::conversion::utils
