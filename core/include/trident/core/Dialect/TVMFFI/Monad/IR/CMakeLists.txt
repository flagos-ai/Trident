# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect(TVMFFIMonad tvm_ffi_monad)
set(LLVM_TARGET_DEFINITIONS TVMFFIMonad.td)
mlir_tablegen(TVMFFITypes.h.inc -gen-typedef-decls
              -typedefs-dialect=tvm_ffi_monad)
mlir_tablegen(TVMFFITypes.cpp.inc -gen-typedef-defs
              -typedefs-dialect=tvm_ffi_monad)
add_public_tablegen_target(MLIRTVMFFIMonadTypeDefsIncGen)
add_dependencies(MLIRTVMFFIMonadIncGen MLIRTVMFFIMonadTypeDefsIncGen)
add_dependencies(MLIRTVMFFIMonadIncGen MLIRTVMFFITypeInterfacesIncGen)
