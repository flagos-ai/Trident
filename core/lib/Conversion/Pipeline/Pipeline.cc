//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Pipeline/Pipeline.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "trident/core/Conversion/TVMFFIABIToFunc/TVMFFIABIToFunc.h"
#include "trident/core/Conversion/TVMFFIABIToGPU/TVMFFIABIToGPU.h"
#include "trident/core/Conversion/TVMFFIABIToLLVM/TVMFFIABIToLLVM.h"
#include "trident/core/Conversion/TVMFFIAPIToABI/TVMFFIAPIToABI.h"
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/LowerTVMFFIABIDefer.h"
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/LowerTVMFFIABITensorMetadata.h"
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/LowerTVMFFIABIToPack.h"
#include "trident/core/Dialect/TVMFFI/API/Transforms/LowerTVMFFIAPIKernelLaunch.h"
#include "trident/core/Dialect/TVMFFI/Arith/Transforms/LowerTVMFFIArith.h"
#include "trident/core/Dialect/TVMFFI/Monad/Transforms/LowerTVMFFIMonad.h"
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/Passes.h> // NOLINT(misc-include-cleaner)
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>     // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/SCF/IR/SCF.h>         // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h> // NOLINT(misc-include-cleaner)
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

namespace trident::conversion {

#define GEN_PASS_DEF_TRIDENTLOWERINGPIPELINE
#include "trident/core/Conversion/Passes.h.inc"

class TridentLoweringPipelinePass final
    : public impl::TridentLoweringPipelineBase<TridentLoweringPipelinePass> {
  void runOnOperation() final {
    mlir::PassManager pm(&getContext(), mlir::ModuleOp::getOperationName());
    // Expand semantic arithmetic and guarded launches before simplifying
    // Results.
    pm.addPass(tvm_ffi::arith::createLowerTVMFFIArith());
    pm.addPass(tvm_ffi::api::createLowerTVMFFIAPIKernelLaunch());
    pm.addPass(createConvertTVMFFIABIToGPU());
    // Simplify continuations while Result success and payloads are explicit.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(tvm_ffi::monad::createLowerTVMFFIMonad());
    pm.addPass(mlir::createCanonicalizerPass());
    // Materialize lexical cleanup before inlining can merge its block scopes.
    pm.addPass(trident::conversion::createConvertTVMFFIAPIToABI());
    pm.addPass(tvm_ffi::abi::createLowerTVMFFIABIDefer());
    // Remove unused ABI declarations before packed entries take their names.
    pm.addPass(mlir::createSymbolDCEPass());
    // Form packed entry bodies before lowering their native ABI.
    pm.addPass(tvm_ffi::abi::createLowerTVMFFIABITensorMetadata());
    pm.addPass(tvm_ffi::abi::createLowerTVMFFIABIToPack());
    pm.addPass(createConvertTVMFFIABIToFunc());
    pm.addPass(mlir::createCanonicalizerPass());
    // Packed ABI preflight introduces CFG blocks. Structure them so calls in
    // single-block SCF regions can inline the entire callee, including guards.
    pm.addPass(mlir::createLiftControlFlowToSCFPass());
    // Local calls now target packed ABI functions. Inline while their control
    // flow still uses SCF regions, before lowering SCF to the LLVM CFG.
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    // Expose identical inlined guard predicates to SCF condition propagation.
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    // Clean up forwarding blocks and join arguments before GPU preparation.
    pm.addPass(mlir::createCanonicalizerPass());
    // GPU conversion simplifies regions before applying LLVM patterns. Run it
    // before packed calls and reference checks expand into large LLVM CFGs.
    pm.addPass(
        mlir::createGpuToLLVMConversionPass()); // NOLINT(misc-include-cleaner)
    pm.addPass(createConvertTVMFFIABIToLLVM());
    pm.addPass(createConvertDLPackToLLVM());
    pm.addPass(mlir::createUBToLLVMConversionPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    // Fold the final LLVM IR after removing conversion bridges, then drop
    // symbols made unused by conversion and simplification.
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    if (mlir::failed(pm.run(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::conversion
