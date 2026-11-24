# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect_library(
  TVMFFIMonadDialect
  TVMFFIDialect.cc
  TVMFFIOps.cc
  TVMFFITypes.cc
  DEPENDS
  MLIRTVMFFIMonadIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  TVMFFIAPIDialect
  TVMFFISupport)
