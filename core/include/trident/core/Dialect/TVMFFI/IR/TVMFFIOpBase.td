// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPBASE_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPBASE_TD_

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.td"

class TVMFFI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVMFFI_Dialect, mnemonic, traits>;

#endif
