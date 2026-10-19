//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_H_

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypeInterfaces.h.inc"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

namespace mlir::TypeTrait {

/// Requires bool contains(TVMFFICABIType) const to test C ABI containment.
template <typename ConcreteType>
class Contains : public ::mlir::TypeTrait::TraitBase<ConcreteType, Contains> {};

/// Marks definite C ABI object types.
template <typename ConcreteType>
class HasObject : public ::mlir::TypeTrait::TraitBase<ConcreteType, HasObject> {
};

/// Marks a type exposing a constexpr getTypeIndex() accessor.
template <typename ConcreteType>
class HasStaticTypeIndex
    : public ::mlir::TypeTrait::TraitBase<ConcreteType, HasStaticTypeIndex> {};

template <typename ConcreteType>
class TVMFFIABI : public ::mlir::TypeTrait::TraitBase<ConcreteType, TVMFFIABI> {
};

template <typename ConcreteType>
class TVMFFICABI
    : public ::mlir::TypeTrait::TraitBase<ConcreteType, TVMFFICABI> {};

template <typename ConcreteType>
class TVMFFICBaseABI
    : public ::mlir::TypeTrait::TraitBase<ConcreteType, TVMFFICBaseABI> {};

template <typename ConcreteType>
class TVMFFICOpaqueABI
    : public ::mlir::TypeTrait::TraitBase<ConcreteType, TVMFFICOpaqueABI> {};

} // namespace mlir::TypeTrait

namespace trident::tvm_ffi {

class TVMFFIABIType : public ::mlir::Type {
public:
  using ::mlir::Type::Type;

  bool isObject() const;

  static bool classof(::mlir::Type type);
  static ::mlir::Value getZero(::mlir::OpBuilder &builder, ::mlir::Location loc,
                               TVMFFIABIType type);
  static bool isZero(::mlir::Value value);
  static ::mlir::Value load(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::Value slot);
};

/// Common base for concrete and opaque semantic TVM FFI ABI types.
class TVMFFICABIType : public TVMFFIABIType {
public:
  using TVMFFIABIType::TVMFFIABIType;

  static bool classof(::mlir::Type type);
  static ::mlir::LLVM::LLVMStructType getLLVMType(::mlir::MLIRContext *context);
};

/// A type with a public C representation or object layout and a type index.
class TVMFFICBaseABIType : public TVMFFICABIType {
public:
  using TVMFFICABIType::TVMFFICABIType;

  int32_t getTypeIndex() const;
  static bool classof(::mlir::Type type);
};

/// A semantic ABI type requiring decomposition or specialized lowering.
class TVMFFICOpaqueABIType : public TVMFFICABIType {
public:
  using TVMFFICABIType::TVMFFICABIType;

  static bool classof(::mlir::Type type);
};

/// A HasObject classification view; concrete types inherit the Base ABI class.
class TVMFFIObjectType : public TVMFFICBaseABIType {
public:
  using TVMFFICBaseABIType::TVMFFICBaseABIType;

  static bool classof(::mlir::Type type);
};

} // namespace trident::tvm_ffi

#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h.inc"

#endif
