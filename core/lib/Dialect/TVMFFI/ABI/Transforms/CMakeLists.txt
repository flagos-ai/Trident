# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_library(
  TVMFFIABITransforms
  LowerTVMFFIABIDefer.cc
  LowerTVMFFIABITensorMetadata.cc
  LowerTVMFFIABIToPack.cc
  DEPENDS
  MLIRTVMFFIABITransformsPassIncGen
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
  TVMFFIABIDialect)
