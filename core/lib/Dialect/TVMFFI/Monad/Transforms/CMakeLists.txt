# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_library(
  TVMFFIMonadTransforms
  LowerTVMFFIMonad.cc
  DEPENDS
  MLIRTVMFFIMonadTransformsPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRControlFlowTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransformUtils
  MLIRTransforms
  TVMFFIAPIDialect
  TVMFFIMonadDialect)
