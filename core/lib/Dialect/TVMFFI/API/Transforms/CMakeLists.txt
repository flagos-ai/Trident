# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_library(
  TVMFFIAPITransforms
  LowerTVMFFIAPIKernelLaunch.cc
  DEPENDS
  MLIRTVMFFIAPITransformsPassIncGen
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
