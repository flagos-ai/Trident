# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_conversion_library(
  TVMFFIAPIToABI
  TVMFFIAPIToABI.cc
  DEPENDS
  MLIRConversionPassIncGen
  LINK_LIBS
  PUBLIC
  DLPackDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRControlFlowTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransformUtils
  MLIRTransforms
  TVMFFIABIDialect
  TVMFFIAPIDialect
  TVMFFIMonadDialect)
