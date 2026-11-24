//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFITypes.h"
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace trident::tvm_ffi {

mlir::Value TVMFFIABIType::getZero(mlir::OpBuilder &builder, mlir::Location loc,
                                   TVMFFIABIType type) {
  mlir::Type const storage =
      mlir::isa<abi::FunctionType, abi::ObjectType>(type)
          ? mlir::Type(mlir::LLVM::LLVMPointerType::get(builder.getContext()))
          : mlir::Type(TVMFFICABIType::getLLVMType(builder.getContext()));
  mlir::Value const zero = mlir::LLVM::ZeroOp::create(builder, loc, storage);
  return mlir::UnrealizedConversionCastOp::create(builder, loc, type, zero)
      .getResult(0);
}

bool TVMFFIABIType::isZero(mlir::Value value) {
  mlir::UnrealizedConversionCastOp cast =
      value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  return cast && cast.getNumOperands() == 1 && cast.getNumResults() == 1 &&
         mlir::isa<TVMFFIABIType>(value.getType()) &&
         (cast.getOperand(0).getType() ==
              TVMFFICABIType::getLLVMType(value.getContext()) ||
          (mlir::isa<abi::FunctionType, abi::ObjectType>(value.getType()) &&
           mlir::isa<mlir::LLVM::LLVMPointerType>(
               cast.getOperand(0).getType()))) &&
         cast.getOperand(0).getDefiningOp<mlir::LLVM::ZeroOp>();
}

mlir::Value TVMFFIABIType::load(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value slot) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(context, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(context);
  mlir::LLVM::LLVMStructType const anyTy = TVMFFICABIType::getLLVMType(context);
  mlir::Value const payloadPtr =
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
  return mlir::LLVM::LoadOp::create(builder, loc, i64Ty, payloadPtr);
}

} // namespace trident::tvm_ffi
