//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/Support/TVMFFITypes.h"
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

namespace trident::tvm_ffi {

bool TVMFFIABIType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::TVMFFIABI>();
}

bool TVMFFIABIType::isObject() const {
  return static_cast<mlir::Type>(*this).hasTrait<mlir::TypeTrait::HasObject>();
}

bool TVMFFICABIType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::TVMFFICABI>();
}

mlir::LLVM::LLVMStructType
TVMFFICABIType::getLLVMType(mlir::MLIRContext *context) {
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(context, 64);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i64Ty});
}

bool TVMFFICBaseABIType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::TVMFFICBaseABI>();
}

bool TVMFFICOpaqueABIType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::TVMFFICOpaqueABI>();
}

bool TVMFFIIntegralType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::IsIntegral>();
}

bool TVMFFINumericType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::IsNumeric>();
}

bool TVMFFIObjectType::classof(mlir::Type type) {
  return type && type.hasTrait<mlir::TypeTrait::HasObject>();
}

} // namespace trident::tvm_ffi
