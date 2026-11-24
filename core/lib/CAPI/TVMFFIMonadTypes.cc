//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFIMonadTypes.h"
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFITypes.h"
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

MlirType mlirTVMFFIMonadResultTypeGetChecked(MlirLocation location,
                                             MlirType payload) {
  mlir::Location const loc = unwrap(location);
  trident::tvm_ffi::TVMFFICABIType const type =
      mlir::dyn_cast_or_null<trident::tvm_ffi::TVMFFICABIType>(unwrap(payload));
  if (!type) {
    mlir::emitError(loc) << "Result payload must be a TVM FFI C ABI type";
    return MlirType{nullptr};
  } else if (type.getContext() != loc.getContext()) {
    mlir::emitError(loc)
        << "Result payload must belong to the construction context";
    return MlirType{nullptr};
  } else {
    return wrap(trident::tvm_ffi::monad::ResultType::getChecked(
        loc, loc.getContext(), type));
  }
}

MlirStringRef mlirTVMFFIMonadResultTypeGetName(void) {
  return wrap(trident::tvm_ffi::monad::ResultType::name);
}

MlirType mlirTVMFFIMonadResultTypeGetPayloadType(MlirType type) {
  return wrap(mlir::cast<trident::tvm_ffi::monad::ResultType>(unwrap(type))
                  .getPayloadType());
}

MlirTypeID mlirTVMFFIMonadResultTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::monad::ResultType::getTypeID());
}

bool mlirTypeIsATVMFFIMonadResultType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::monad::ResultType>(unwrap(type));
}
