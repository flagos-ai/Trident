//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// TVMFFIDialect.cc - TVMFFI Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TVMFFI.td via ODS-generated .cpp.inc files.
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFITypes.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>
#include <string>

#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIABIDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIABI.cpp.inc"

namespace trident::tvm_ffi::abi {

struct TVMFFIInlinerInterface final : mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }
};

static bool isTaggedCastType(mlir::Type type) {
  return mlir::isa<AnyType>(type) ||
         (mlir::isa<TVMFFICBaseABIType>(type) &&
          mlir::isa<TVMFFITypeIndexInterface>(type));
}
bool DowncastOp::areCastCompatible(mlir::TypeRange inputs,
                                   mlir::TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1 &&
         isTaggedCastType(inputs[0]) && isTaggedCastType(outputs[0]) &&
         (inputs[0] == outputs[0] || mlir::isa<AnyType>(inputs[0]));
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

llvm::ArrayRef<mlir::Type> FuncOp::getArgumentTypes() {
  return getFunctionType().getInputs();
}

// FuncOp custom assembly format.

mlir::Region *FuncOp::getCallableRegion() {
  mlir::Region &body = getBody();
  return body.empty() ? nullptr : &body;
}

llvm::ArrayRef<mlir::Type> FuncOp::getResultTypes() {
  return getFunctionType().getResults();
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name),
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) -> mlir::FunctionType {
        return builder.getFunctionType(argTypes, results);
      },
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void PackFuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, mlir::FunctionType type,
                       llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

llvm::ArrayRef<mlir::Type> PackFuncOp::getArgumentTypes() {
  return getFunctionType().getInputs();
}

// PackFuncOp custom assembly format.

mlir::Region *PackFuncOp::getCallableRegion() {
  mlir::Region &body = getBody();
  return body.empty() ? nullptr : &body;
}

llvm::ArrayRef<mlir::Type> PackFuncOp::getResultTypes() {
  return getFunctionType().getResults();
}

mlir::ParseResult PackFuncOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name),
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) -> mlir::FunctionType {
        return builder.getFunctionType(argTypes, results);
      },
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void PackFuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void TVMFFIABIDialect::initialize() {
  addInterfaces<TVMFFIInlinerInterface>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIABI.cpp.inc"
      >();
}

mlir::Operation *
TVMFFIABIDialect::materializeConstant(mlir::OpBuilder &builder,
                                      mlir::Attribute value, mlir::Type type,
                                      mlir::Location location) {
  return mlir::arith::ConstantOp::isBuildableWith(value, type)
             ? mlir::arith::ConstantOp::create(
                   builder, location, mlir::cast<mlir::TypedAttr>(value))
             : nullptr;
}

bool UpcastOp::areCastCompatible(mlir::TypeRange inputs,
                                 mlir::TypeRange outputs) {
  return inputs.size() == 1 && outputs.size() == 1 &&
         isTaggedCastType(inputs[0]) && isTaggedCastType(outputs[0]) &&
         (inputs[0] == outputs[0] || mlir::isa<AnyType>(outputs[0]));
}

} // namespace trident::tvm_ffi::abi
