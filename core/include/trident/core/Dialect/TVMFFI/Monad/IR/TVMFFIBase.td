// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_TVMFFI_MONAD_BASE_TD
#define TRIDENT_TVMFFI_MONAD_BASE_TD
include "mlir/IR/DialectBase.td"
def TVMFFIMonad_Dialect : Dialect {
  let name = "tvm_ffi_monad";
  let cppNamespace = "::trident::tvm_ffi::monad";
  let dependentDialects = ["::trident::tvm_ffi::api::TVMFFIAPIDialect"];
  let summary = "TVM FFI monad dialect";
  let useDefaultTypePrinterParser = 1;
}
#endif
