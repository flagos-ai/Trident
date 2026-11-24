# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

add_mlir_pdll_library(
  TVMFFIABIToLLVMPDLLPatternsIncGen
  TVMFFIABIToLLVM.pdll
  TVMFFIABIToLLVMPDLLPatterns.h.inc
  EXTRA_INCLUDES
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
  ${LLVM_MAIN_SRC_DIR}/include
  ${LLVM_BINARY_DIR}/include)

add_mlir_conversion_library(
  TVMFFIABIToLLVM
  TVMFFIABIToLLVM.cc
  DEPENDS
  MLIRConversionPassIncGen
  TVMFFIABIToLLVMPDLLPatternsIncGen
  LINK_LIBS
  PUBLIC
  ConversionUtils
  MLIRControlFlowTransforms
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  TVMFFIABIDialect)

target_link_libraries(TVMFFIABIToLLVM PUBLIC ${TORCH_LIBRARIES})

target_include_directories(TVMFFIABIToLLVM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
