# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_conversion_library(
  TVMFFIABIToFunc
  TVMFFIABIToFunc.cc
  DEPENDS
  MLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  TVMFFIABIDialect)
