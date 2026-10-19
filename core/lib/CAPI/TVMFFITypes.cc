//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

#define DEFINE_GENERIC_TYPE(Name)                                              \
  MlirType mlirTVMFFI##Name##TypeGet(MlirContext context) {                    \
    return wrap(trident::tvm_ffi::Name##Type::get(unwrap(context)));           \
  }                                                                            \
  MlirStringRef mlirTVMFFI##Name##TypeGetName(void) {                          \
    return wrap(trident::tvm_ffi::Name##Type::name);                           \
  }                                                                            \
  MlirTypeID mlirTVMFFI##Name##TypeGetTypeID(void) {                           \
    return wrap(trident::tvm_ffi::Name##Type::getTypeID());                    \
  }                                                                            \
  bool mlirTypeIsATVMFFI##Name##Type(MlirType type) {                          \
    return mlir::isa<trident::tvm_ffi::Name##Type>(unwrap(type));              \
  }

DEFINE_GENERIC_TYPE(Any)
DEFINE_GENERIC_TYPE(Bool)
DEFINE_GENERIC_TYPE(CArray)
DEFINE_GENERIC_TYPE(DType)
DEFINE_GENERIC_TYPE(Device)
DEFINE_GENERIC_TYPE(Exception)
DEFINE_GENERIC_TYPE(Float)
DEFINE_GENERIC_TYPE(Function)
DEFINE_GENERIC_TYPE(Int)
DEFINE_GENERIC_TYPE(None)
DEFINE_GENERIC_TYPE(Object)
DEFINE_GENERIC_TYPE(RawStr)
DEFINE_GENERIC_TYPE(SmallStr)
DEFINE_GENERIC_TYPE(Str)
DEFINE_GENERIC_TYPE(Tensor)

#undef DEFINE_GENERIC_TYPE

MlirType mlirTVMFFIArrayTypeGetChecked(MlirLocation location,
                                       MlirType element) {
  mlir::Location const loc = unwrap(location);
  mlir::Type const type = unwrap(element);
  if (!type || type.getContext() != loc.getContext()) {
    mlir::emitError(loc)
        << "array element must belong to the construction context";
    return MlirType{nullptr};
  }
  trident::tvm_ffi::TVMFFICABIType const elementType =
      mlir::dyn_cast<trident::tvm_ffi::TVMFFICABIType>(type);
  if (!elementType) {
    mlir::emitError(loc) << "array element must be a TVM FFI C ABI type";
    return MlirType{nullptr};
  }
  return wrap(trident::tvm_ffi::ArrayType::getChecked(loc, loc.getContext(),
                                                      elementType));
}

MlirType mlirTVMFFIArrayTypeGetElementType(MlirType type) {
  return wrap(
      mlir::cast<trident::tvm_ffi::ArrayType>(unwrap(type)).getElementType());
}

MlirStringRef mlirTVMFFIArrayTypeGetName(void) {
  return wrap(trident::tvm_ffi::ArrayType::name);
}

MlirTypeID mlirTVMFFIArrayTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::ArrayType::getTypeID());
}

bool mlirTypeIsATVMFFIArrayType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::ArrayType>(unwrap(type));
}

MlirType mlirTVMFFIResultTypeGetChecked(MlirLocation location,
                                        MlirType payload) {
  mlir::Location const loc = unwrap(location);
  const mlir::Type type = unwrap(payload);
  if (!type) {
    mlir::emitError(loc) << "Result requires one payload type";
    return MlirType{nullptr};
  }
  if (type.getContext() != loc.getContext()) {
    mlir::emitError(loc)
        << "Result payload must belong to the construction context";
    return MlirType{nullptr};
  }
  return wrap(
      trident::tvm_ffi::ResultType::getChecked(loc, loc.getContext(), type));
}

MlirStringRef mlirTVMFFIResultTypeGetName(void) {
  return wrap(trident::tvm_ffi::ResultType::name);
}

MlirType mlirTVMFFIResultTypeGetPayloadType(MlirType type) {
  return wrap(
      mlir::cast<trident::tvm_ffi::ResultType>(unwrap(type)).getPayloadType());
}

MlirTypeID mlirTVMFFIResultTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::ResultType::getTypeID());
}

bool mlirTypeIsATVMFFIResultType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::ResultType>(unwrap(type));
}

MlirType mlirTVMFFITupleTypeGetChecked(MlirLocation location, intptr_t numTypes,
                                       const MlirType *types) {
  mlir::Location const loc = unwrap(location);
  if (numTypes < 0 || (numTypes > 0 && !types)) {
    mlir::emitError(loc) << "tuple requires a valid member array";
    return MlirType{nullptr};
  }
  llvm::SmallVector<trident::tvm_ffi::TVMFFICABIType> members;
  members.reserve(numTypes);
  for (intptr_t index = 0; index < numTypes; ++index) {
    const mlir::Type member = unwrap(types[index]);
    if (!member || member.getContext() != loc.getContext()) {
      mlir::emitError(loc)
          << "tuple members must belong to the construction context";
      return MlirType{nullptr};
    }
    trident::tvm_ffi::TVMFFICABIType const cabi =
        mlir::dyn_cast<trident::tvm_ffi::TVMFFICABIType>(member);
    if (!cabi) {
      mlir::emitError(loc) << "tuple elements must be TVM FFI C ABI types";
      return MlirType{nullptr};
    }
    members.push_back(cabi);
  }
  return wrap(trident::tvm_ffi::TupleType::getChecked(
      loc, loc.getContext(),
      llvm::ArrayRef<trident::tvm_ffi::TVMFFICABIType>(members)));
}

MlirStringRef mlirTVMFFITupleTypeGetName(void) {
  return wrap(trident::tvm_ffi::TupleType::name);
}

intptr_t mlirTVMFFITupleTypeGetNumTypes(MlirType type) {
  return static_cast<intptr_t>(
      mlir::cast<trident::tvm_ffi::TupleType>(unwrap(type)).getTypes().size());
}

MlirType mlirTVMFFITupleTypeGetType(MlirType type, intptr_t index) {
  return wrap(
      mlir::cast<trident::tvm_ffi::TupleType>(unwrap(type)).getTypes()[index]);
}

MlirTypeID mlirTVMFFITupleTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::TupleType::getTypeID());
}

bool mlirTypeIsATVMFFITupleType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::TupleType>(unwrap(type));
}

MlirType mlirTVMFFIUnionTypeGetChecked(MlirLocation location, intptr_t numTypes,
                                       const MlirType *types) {
  mlir::Location const loc = unwrap(location);
  if (numTypes < 0 || (numTypes > 0 && !types)) {
    mlir::emitError(loc) << "union requires a valid member array";
    return MlirType{nullptr};
  }
  llvm::SmallVector<trident::tvm_ffi::TVMFFICABIType> members;
  members.reserve(numTypes);
  for (intptr_t index = 0; index < numTypes; ++index) {
    const mlir::Type member = unwrap(types[index]);
    if (!member || member.getContext() != loc.getContext()) {
      mlir::emitError(loc)
          << "union members must belong to the construction context";
      return MlirType{nullptr};
    }
    trident::tvm_ffi::TVMFFICABIType const cabi =
        mlir::dyn_cast<trident::tvm_ffi::TVMFFICABIType>(member);
    if (!cabi) {
      mlir::emitError(loc) << "union members must be TVM FFI C ABI types";
      return MlirType{nullptr};
    }
    members.push_back(cabi);
  }
  return wrap(trident::tvm_ffi::UnionType::getChecked(
      loc, loc.getContext(),
      llvm::ArrayRef<trident::tvm_ffi::TVMFFICABIType>(members)));
}

MlirStringRef mlirTVMFFIUnionTypeGetName(void) {
  return wrap(trident::tvm_ffi::UnionType::name);
}

intptr_t mlirTVMFFIUnionTypeGetNumTypes(MlirType type) {
  return static_cast<intptr_t>(
      mlir::cast<trident::tvm_ffi::UnionType>(unwrap(type)).getTypes().size());
}

MlirType mlirTVMFFIUnionTypeGetType(MlirType type, intptr_t index) {
  return wrap(
      mlir::cast<trident::tvm_ffi::UnionType>(unwrap(type)).getTypes()[index]);
}

MlirTypeID mlirTVMFFIUnionTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::UnionType::getTypeID());
}

bool mlirTypeIsATVMFFIUnionType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::UnionType>(unwrap(type));
}
