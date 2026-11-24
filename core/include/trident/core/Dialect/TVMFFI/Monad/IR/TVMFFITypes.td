// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_TVMFFI_MONAD_TYPES_TD
#define TRIDENT_TVMFFI_MONAD_TYPES_TD
include "mlir/IR/AttrTypeBase.td"
include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/Support/TVMFFIInterfaces.td"
def TVMFFIMonad_ResultType
    : TypeDef<TVMFFIMonad_Dialect, "Result", [TVMFFI_ResultTypeInterface]> {
  let mnemonic = "result";
  let summary = "One TVM FFI C ABI payload guarded by a success bit";
  let parameters = (ins "::trident::tvm_ffi::TVMFFICABIType":$payloadType);
  let assemblyFormat = "`<` $payloadType `>`";
  let genVerifyDecl = 1;
}

#endif
