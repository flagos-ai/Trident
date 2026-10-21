// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_MONAD_TVMFFITYPES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_MONAD_TVMFFITYPES_TD_

include "trident/core/Dialect/TVMFFI/IR/TVMFFITypeBase.td"

def TVMFFI_ResultType
    : TVMFFI_Type<"Result", "monad.result"> {
  let summary = "One TVM FFI C ABI payload guarded by a success bit";
  let parameters = (ins "::trident::tvm_ffi::TVMFFICABIType":$payloadType);
  let assemblyFormat = "`<` $payloadType `>`";
  let genVerifyDecl = 1;
}

#endif
