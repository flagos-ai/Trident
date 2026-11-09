//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace trident::tvm_ffi {

bool AnyType::contains(TVMFFICABIType) const { return true; }

mlir::LogicalResult
ArrayType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  TVMFFICABIType element) {
  return TVMFFICABIType::classof(element)
             ? mlir::success()
             : emitError() << "array element must be a TVM FFI C ABI type";
}

mlir::Type FunctionType::getNativeType() const {
  return ObjectType::get(getContext());
}

mlir::Type TensorType::getNativeType() const {
  return ObjectType::get(getContext());
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
        mlir::isa<AnyType, CAnyType, UnionType>(type)) {
      return emitError() << "invalid union member type " << type;
    }
    if (!mlir::isa<TVMFFITypeIndexInterface, ExceptionType, FunctionType,
                   StrType, TensorType>(type)) {
      return emitError() << "invalid union member type " << type;
    }
    if (!uniqueTypes.insert(type).second) {
      return emitError() << "union member types must be unique";
    }
  }
  return mlir::success();
}

} // namespace trident::tvm_ffi
