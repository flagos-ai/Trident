//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_ABI_IR_TVMFFIBASE_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_ABI_IR_TVMFFIBASE_TD_

include "mlir/IR/DialectBase.td"

def TVMFFIABI_Dialect : Dialect {
  let name = "tvm_ffi_abi";
  let cppNamespace = "::trident::tvm_ffi::abi";
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::LLVM::LLVMDialect"];
  let hasConstantMaterializer = 1;
  let summary = "TVM FFI ABI dialect";
  let description = [{
    The TVM FFI ABI dialect models the TVM runtime FFI (foreign function interface)
    for interoperability between TVM/PyTorch and Triton-compiled kernels.
    This dialect provides operations for managing TVM-packed arguments,
    invoking TVM runtime functions, and converting between TVM and native data
    types.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 1;
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_ABI_IR_TVMFFIBASE_TD_
