# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_interface(TVMFFIInterfaces)
set(LLVM_TARGET_DEFINITIONS TVMFFIInterfaces.td)
mlir_tablegen(TVMFFITypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TVMFFITypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRTVMFFITypeInterfacesIncGen)
