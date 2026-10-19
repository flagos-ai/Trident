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
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
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

namespace {

bool isTaggedCastType(mlir::Type type) {
  if (mlir::isa<AnyType, ArrayType, TupleType>(type)) {
    return true;
  }
  if (UnionType const unionType = mlir::dyn_cast<UnionType>(type)) {
    return llvm::all_of(unionType.getTypes(), isTaggedCastType);
  }
  return !mlir::isa<FunctionType>(type) && mlir::isa<TVMFFICABIType>(type) &&
         mlir::isa<TVMFFITypeIndexInterface>(type);
}

} // namespace

mlir::LogicalResult
ArrayType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  TVMFFICABIType element) {
  return TVMFFICABIType::classof(element)
             ? mlir::success()
             : emitError() << "array element must be a TVM FFI C ABI type";
}

bool DowncastOp::areCastCompatible(mlir::TypeRange inputs,
                                   mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1 ||
      !isTaggedCastType(inputs[0]) || !isTaggedCastType(outputs[0]) ||
      mlir::isa<AnyType>(outputs[0]) || inputs[0] == outputs[0]) {
    return false;
  }
  if (mlir::isa<AnyType>(inputs[0])) {
    return true;
  }
  UnionType const source = mlir::dyn_cast<UnionType>(inputs[0]);
  if (!source || !source.contains(mlir::cast<TVMFFICABIType>(outputs[0]))) {
    return false;
  }
  UnionType const target = mlir::dyn_cast<UnionType>(outputs[0]);
  return !target || !target.contains(mlir::cast<TVMFFICABIType>(inputs[0]));
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

mlir::LogicalResult
ResultType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   TVMFFICABIType payload) {
  return TVMFFICABIType::classof(payload)
             ? mlir::success()
             : emitError() << "Result payload must be a TVM FFI C ABI type";
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

mlir::Type TupleType::parse(mlir::AsmParser &parser) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  llvm::SmallVector<TVMFFICABIType> types;
  if (mlir::failed(parser.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::LessGreater, [&]() -> mlir::ParseResult {
            mlir::Type type;
            if (mlir::failed(parser.parseType(type))) {
              return mlir::failure();
            }
            TVMFFICABIType const member = mlir::dyn_cast<TVMFFICABIType>(type);
            if (!member) {
              return parser.emitError(loc)
                     << "tuple elements must be TVM FFI C ABI types";
            }
            types.push_back(member);
            return mlir::success();
          }))) {
    return {};
  }
  return getChecked(
      [&]() -> mlir::InFlightDiagnostic { return parser.emitError(loc); },
      parser.getContext(), llvm::ArrayRef<TVMFFICABIType>(types));
}

void TupleType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  llvm::interleaveComma(getTypes(), printer);
  printer << '>';
}

mlir::LogicalResult
TupleType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<TVMFFICABIType> types) {
  return llvm::all_of(types,
                      [](TVMFFICABIType type) -> bool {
                        return TVMFFICABIType::classof(type);
                      })
             ? mlir::success()
             : emitError() << "tuple elements must be TVM FFI C ABI types";
}

bool UnionType::contains(TVMFFICABIType type) const {
  if (const UnionType unionType = mlir::dyn_cast<UnionType>(type)) {
    return llvm::all_of(unionType.getTypes(),
                        [&](const mlir::Type member) -> bool {
                          return llvm::is_contained(getTypes(), member);
                        });
  }
  return llvm::is_contained(getTypes(), type);
}

mlir::Type UnionType::parse(mlir::AsmParser &parser) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  llvm::SmallVector<TVMFFICABIType> types;
  if (mlir::failed(parser.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::LessGreater, [&]() -> mlir::ParseResult {
            mlir::Type type;
            if (mlir::failed(parser.parseType(type))) {
              return mlir::failure();
            }
            TVMFFICABIType const member = mlir::dyn_cast<TVMFFICABIType>(type);
            if (!member) {
              return parser.emitError(loc)
                     << "union members must be TVM FFI C ABI types";
            }
            types.push_back(member);
            return mlir::success();
          }))) {
    return {};
  }
  return getChecked(
      [&]() -> mlir::InFlightDiagnostic { return parser.emitError(loc); },
      parser.getContext(), llvm::ArrayRef<TVMFFICABIType>(types));
}

void UnionType::print(mlir::AsmPrinter &printer) const {
  printer << '<';
  llvm::interleaveComma(getTypes(), printer);
  printer << '>';
}

mlir::LogicalResult
UnionType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<TVMFFICABIType> types) {
  if (types.size() < 2) {
    return emitError() << "union requires at least two member types";
  }
  llvm::SmallDenseSet<mlir::Type> uniqueTypes;
  for (mlir::Type const type : types) {
    if (!mlir::isa<TVMFFICABIType>(type) ||
        mlir::isa<AnyType, UnionType>(type)) {
      return emitError() << "invalid union member type " << type;
    }
    if (!mlir::isa<TVMFFITypeIndexInterface>(type)) {
      return emitError() << "invalid union member type " << type;
    }
    if (!uniqueTypes.insert(type).second) {
      return emitError() << "union member types must be unique";
    }
  }
  return mlir::success();
}

bool UpcastOp::areCastCompatible(mlir::TypeRange inputs,
                                 mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }
  mlir::Type const input = inputs[0];
  mlir::Type output = outputs[0];
  return mlir::isa<AnyType, UnionType>(output) && isTaggedCastType(input) &&
         isTaggedCastType(output) &&
         (mlir::isa<AnyType>(output) || mlir::cast<UnionType>(output).contains(
                                            mlir::cast<TVMFFICABIType>(input)));
}

} // namespace trident::tvm_ffi
