// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_TVMFFI_ARITH_BASE_TD
#define TRIDENT_TVMFFI_ARITH_BASE_TD
include "mlir/IR/DialectBase.td"
def TVMFFIArith_Dialect : Dialect {
  let name = "tvm_ffi_arith";
  let cppNamespace = "::trident::tvm_ffi::arith";
  let dependentDialects = ["::trident::tvm_ffi::abi::TVMFFIABIDialect"];
  let summary = "TVM FFI arith dialect";
}
#endif
