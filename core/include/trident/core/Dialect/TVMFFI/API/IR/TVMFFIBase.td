//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFIBASE_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFIBASE_TD_

include "mlir/IR/DialectBase.td"

def TVMFFIAPI_Dialect : Dialect {
  let name = "tvm_ffi_api";
  let cppNamespace = "::trident::tvm_ffi::api";
  let dependentDialects = ["::trident::tvm_ffi::abi::TVMFFIABIDialect", "::mlir::arith::ArithDialect",
                           "::mlir::LLVM::LLVMDialect"];
  let hasConstantMaterializer = 1;
  let summary = "TVM FFI API dialect";
  let description = [{
    The TVM FFI API dialect models the TVM runtime FFI (foreign function interface)
    for interoperability between TVM/PyTorch and Triton-compiled kernels.
    This dialect provides operations for managing TVM-packed arguments,
    invoking TVM runtime functions, and converting between TVM and native data
    types.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_API_IR_TVMFFIBASE_TD_
