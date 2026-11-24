# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_conversion_library(
  TVMFFIABIToGPU
  TVMFFIABIToGPU.cc
  DEPENDS
  MLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRGPUDialect
  MLIRPass
  MLIRSCFDialect
  TVMFFIABIDialect)
