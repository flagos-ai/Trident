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

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>             // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>
#include <string>

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

namespace trident::tvm_ffi {

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
  if (mlir::isa<AnyType, CAnyType, ArrayType, ExceptionType, FunctionType,
                StrType, TensorType, TupleType>(type)) {
    return true;
  }
  if (UnionType const unionType = mlir::dyn_cast<UnionType>(type)) {
    return llvm::all_of(unionType.getTypes(), isTaggedCastType);
  }
  return mlir::isa<TVMFFICABIType>(type) &&
         mlir::isa<TVMFFITypeIndexInterface>(type);
}

bool DowncastOp::areCastCompatible(mlir::TypeRange inputs,
                                   mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1 ||
      !isTaggedCastType(inputs[0]) || !isTaggedCastType(outputs[0])) {
    return false;
  }
  if (inputs[0] == outputs[0]) {
    return true;
  }
  if (mlir::isa<AnyType, CAnyType>(outputs[0])) {
    return false;
  }
  return llvm::TypeSwitch<mlir::Type, bool>(inputs[0])
      .Case<AnyType, CAnyType>([](mlir::Type) -> bool { return true; })
      .Case<StrType>([&](StrType) -> bool {
        return mlir::isa<RawStrType, SmallStrType, CStrType>(outputs[0]);
      })
      .Case<UnionType>([&](UnionType source) -> bool {
        if (!source.contains(mlir::cast<TVMFFICABIType>(outputs[0]))) {
          return false;
        }
        UnionType const target = mlir::dyn_cast<UnionType>(outputs[0]);
        return !target ||
               !target.contains(mlir::cast<TVMFFICABIType>(inputs[0]));
      })
      .Default([](mlir::Type) -> bool { return false; });
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
         std::string &) { return builder.getFunctionType(argTypes, results); },
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

void TVMFFIDialect::initialize() {
  addInterfaces<TVMFFIInlinerInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"
      >();
}

bool UpcastOp::areCastCompatible(mlir::TypeRange inputs,
                                 mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }
  mlir::Type const input = inputs[0];
  mlir::Type output = outputs[0];
  return isTaggedCastType(input) && isTaggedCastType(output) &&
         (input == output || mlir::isa<AnyType, CAnyType>(output) ||
          (mlir::isa<StrType>(output) &&
           mlir::isa<RawStrType, SmallStrType, CStrType>(input)) ||
          (mlir::isa<UnionType>(output) &&
           mlir::cast<UnionType>(output).contains(
               mlir::cast<TVMFFICABIType>(input))));
}

} // namespace trident::tvm_ffi
