//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Utils/Registration.h"
#include "trident/core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "trident/core/Conversion/TVMFFIABIToFunc/TVMFFIABIToFunc.h"
#include "trident/core/Conversion/TVMFFIABIToGPU/TVMFFIABIToGPU.h"
#include "trident/core/Conversion/TVMFFIABIToLLVM/TVMFFIABIToLLVM.h"
#include "trident/core/Conversion/TVMFFIAPIToABI/TVMFFIAPIToABI.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/LowerTVMFFIABIDefer.h"
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/LowerTVMFFIABITensorMetadata.h"
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/LowerTVMFFIABIToPack.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/API/Transforms/LowerTVMFFIAPIKernelLaunch.h"
#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/Arith/Transforms/LowerTVMFFIArith.h"
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/Monad/Transforms/LowerTVMFFIMonad.h"
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>

void trident::conversion::registerAllDialects(mlir::DialectRegistry &registry) {
  // Register base dialects before extensions and external interfaces.
  mlir::registerAllDialects(registry);
  registry.insert<trident::dlpack::DLPackDialect,
                  trident::tvm_ffi::abi::TVMFFIABIDialect,
                  trident::tvm_ffi::api::TVMFFIAPIDialect,
                  trident::tvm_ffi::arith::TVMFFIArithDialect,
                  trident::tvm_ffi::monad::TVMFFIMonadDialect>();
  mlir::registerAllExtensions(registry);
  trident::conversion::registerConvertDLPackToLLVMInterface(registry);
  trident::conversion::registerConvertTVMFFIABIToLLVMInterface(registry);
}

void trident::conversion::registerAllPasses() {
  mlir::registerAllPasses();
  trident::conversion::registerConvertDLPackToLLVMPass();
  trident::conversion::registerConvertTVMFFIABIToFuncPass();
  trident::conversion::registerConvertTVMFFIABIToGPUPass();
  trident::conversion::registerConvertTVMFFIABIToLLVMPass();
  trident::conversion::registerConvertTVMFFIAPIToABIPass();
  trident::conversion::registerTridentLoweringPipelinePass();
  trident::tvm_ffi::abi::registerLowerTVMFFIABIDeferPass();
  trident::tvm_ffi::abi::registerLowerTVMFFIABITensorMetadataPass();
  trident::tvm_ffi::abi::registerLowerTVMFFIABIToPackPass();
  trident::tvm_ffi::api::registerLowerTVMFFIAPIKernelLaunchPass();
  trident::tvm_ffi::arith::registerLowerTVMFFIArithPass();
  trident::tvm_ffi::monad::registerLowerTVMFFIMonadPass();
}
