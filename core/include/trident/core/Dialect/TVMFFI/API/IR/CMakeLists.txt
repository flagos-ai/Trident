# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect(TVMFFIAPI tvm_ffi_api)
set(LLVM_TARGET_DEFINITIONS TVMFFIAPI.td)
mlir_tablegen(TVMFFITypes.h.inc -gen-typedef-decls
              -typedefs-dialect=tvm_ffi_api)
mlir_tablegen(TVMFFITypes.cpp.inc -gen-typedef-defs
              -typedefs-dialect=tvm_ffi_api)
add_public_tablegen_target(MLIRTVMFFIAPITypeDefsIncGen)
set(LLVM_TARGET_DEFINITIONS TVMFFIInterfaces.td)
mlir_tablegen(TVMFFIAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TVMFFIAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRTVMFFIAPIAttrInterfacesIncGen)
set(LLVM_TARGET_DEFINITIONS TVMFFIAttrs.td)
mlir_tablegen(TVMFFIAttrs.h.inc -gen-attrdef-decls
              -attrdefs-dialect=tvm_ffi_api)
mlir_tablegen(TVMFFIAttrs.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=tvm_ffi_api)
add_public_tablegen_target(MLIRTVMFFIAPIAttrDefsIncGen)
add_dependencies(MLIRTVMFFIAPIIncGen MLIRTVMFFIAPITypeDefsIncGen
                 MLIRTVMFFIInterfacesIncGen MLIRTVMFFITypeInterfacesIncGen)
add_dependencies(MLIRTVMFFIAPIIncGen MLIRTVMFFIAPIAttrInterfacesIncGen
                 MLIRTVMFFIAPIAttrDefsIncGen)
