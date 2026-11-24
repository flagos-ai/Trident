# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_library(
  TVMFFIArithTransforms
  LowerTVMFFIArith.cc
  DEPENDS
  MLIRTVMFFIArithTransformsPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRArithDialect
  MLIRTransforms
  TVMFFIArithDialect)
