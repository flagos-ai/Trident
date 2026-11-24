# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect_library(
  TVMFFIABIDialect
  TVMFFIDialect.cc
  TVMFFIOps.cc
  TVMFFIStorage.cc
  TVMFFITypes.cc
  DEPENDS
  MLIRTVMFFIABIIncGen
  MLIRTVMFFIABITypeDefsIncGen
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
  TVMFFISupport
  tvm_ffi::shared)
target_include_directories(TVMFFIABIDialect
                           PRIVATE ${CMAKE_SOURCE_DIR}/../ffi/include)
