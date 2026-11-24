# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_library(
  TVMFFISupport
  TVMFFIInterfaces.cc
  TVMFFITypes.cc
  DEPENDS
  MLIRTVMFFIInterfacesIncGen
  MLIRTVMFFITypeInterfacesIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRLLVMDialect)
