# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect_library(
  TVMFFIAPIDialect
  TVMFFIAttrs.cc
  TVMFFIDialect.cc
  TVMFFIInterfaces.cc
  TVMFFIOps.cc
  TVMFFITypes.cc
  DEPENDS
  MLIRTVMFFIAPIIncGen
  MLIRTVMFFIAPITypeDefsIncGen
  MLIRTVMFFIAPIAttrDefsIncGen
  MLIRTVMFFIAPIAttrInterfacesIncGen
  LINK_LIBS
  PUBLIC
  DLPackDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  TVMFFIABIDialect
  TVMFFISupport
  tvm_ffi::shared)
target_include_directories(TVMFFIAPIDialect
                           PRIVATE ${CMAKE_SOURCE_DIR}/../ffi/include)
