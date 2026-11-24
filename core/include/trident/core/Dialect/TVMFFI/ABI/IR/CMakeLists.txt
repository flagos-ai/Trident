# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect(TVMFFIABI tvm_ffi_abi)
set(LLVM_TARGET_DEFINITIONS TVMFFIABI.td)
mlir_tablegen(TVMFFITypes.h.inc -gen-typedef-decls
              -typedefs-dialect=tvm_ffi_abi)
mlir_tablegen(TVMFFITypes.cpp.inc -gen-typedef-defs
              -typedefs-dialect=tvm_ffi_abi)
add_public_tablegen_target(MLIRTVMFFIABITypeDefsIncGen)
add_dependencies(MLIRTVMFFIABIIncGen MLIRTVMFFIABITypeDefsIncGen
                 MLIRTVMFFIInterfacesIncGen MLIRTVMFFITypeInterfacesIncGen)
