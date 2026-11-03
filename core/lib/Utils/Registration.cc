//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Utils/Registration.h"
#include "trident/core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "trident/core/Conversion/FinalizeTVMFFI/FinalizeTVMFFI.h"
#include "trident/core/Conversion/Pipeline/Pipeline.h"
#include "trident/core/Conversion/TVMFFIToAscend/TVMFFIToAscend.h"
#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h"
#include "trident/core/Conversion/TVMFFIToGPU/TVMFFIToGPU.h"
#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/Transforms/OwnershipDeallocation.h"
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Pass/PassRegistry.h>

void trident::conversion::registerAllDialects(mlir::DialectRegistry &registry) {
  // Register base dialects before extensions and external interfaces.
  mlir::registerAllDialects(registry);
  registry.insert<trident::dlpack::DLPackDialect,
                  trident::tvm_ffi::TVMFFIDialect>();
  mlir::registerAllExtensions(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::gpu::registerConvertGpuToLLVMInterface(registry);
  trident::conversion::registerConvertDLPackToLLVMInterface(registry);
  trident::conversion::registerConvertTVMFFIToLLVMInterface(registry);
  trident::tvm_ffi::registerTVMFFIObjectOwnershipExternalModels(registry);
}

void trident::conversion::registerAllPasses() {
  mlir::registerPass(
      []() { return trident::conversion::createConvertDLPackToLLVM(); });
  mlir::registerPass(
      []() { return trident::conversion::createConvertTVMFFIToAscend(); });
  mlir::registerPass(
      []() { return trident::conversion::createConvertTVMFFIToFunc(); });
  mlir::registerPass(
      []() { return trident::conversion::createConvertTVMFFIToGPU(); });
  mlir::registerPass(
      []() { return trident::conversion::createConvertTVMFFIToLLVM(); });
  mlir::registerPass(
      []() { return trident::conversion::createFinalizeTVMFFI(); });
  mlir::registerPass(
      []() { return trident::conversion::createTridentLoweringPipeline(); });
  mlir::registerPass([]() {
    return trident::conversion::createTridentAscendLoweringPipeline();
  });
  mlir::registerPass(
      []() { return trident::tvm_ffi::createOwnershipDeallocation(); });
}
