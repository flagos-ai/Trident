# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect(TVMFFIArith tvm_ffi_arith)
add_dependencies(MLIRTVMFFIArithIncGen MLIRTVMFFITypeInterfacesIncGen)
