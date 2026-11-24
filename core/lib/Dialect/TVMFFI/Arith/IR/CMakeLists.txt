# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_dialect_library(
  TVMFFIArithDialect
  TVMFFIDialect.cc
  DEPENDS
  MLIRTVMFFIArithIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  TVMFFIABIDialect
  TVMFFISupport)
