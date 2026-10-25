//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToGPU/TVMFFIToGPU.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include <dlpack/dlpack.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <utility>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOGPU
#include "trident/core/Conversion/Passes.h.inc"

/// Convert an unconditional native launch to gpu.launch_func.
class ConvertKernelLaunchOp final
    : public mlir::OpConversionPattern<tvm_ffi::NativeKernelLaunchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::NativeKernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    llvm::SmallVector<mlir::Value> operands(adaptor.getKernelOperands());
    mlir::gpu::KernelDim3 const gridSize{
        adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    mlir::gpu::KernelDim3 const blockSize{adaptor.getBlockSizeX(),
                                          adaptor.getBlockSizeY(),
                                          adaptor.getBlockSizeZ()};

    std::optional<mlir::gpu::KernelDim3> clusterSize;
    if (adaptor.getClusterSizeX() && adaptor.getClusterSizeY() &&
        adaptor.getClusterSizeZ()) {
      clusterSize = mlir::gpu::KernelDim3{adaptor.getClusterSizeX(),
                                          adaptor.getClusterSizeY(),
                                          adaptor.getClusterSizeZ()};
    }

    mlir::Value const dynamicSharedMemorySize =
        adaptor.getDynamicSharedMemorySize();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op->emitOpError("op is not inside a ModuleOp");
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDeviceIndex =
        utils::getOrCreateAOTITorchGetCurrentDeviceIndex(moduleOp);
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getStream =
        utils::getOrCreateTVMFFIEnvGetStream(moduleOp);
    if (mlir::failed(getDeviceIndex) || mlir::failed(getStream)) {
      return op->emitOpError("failed to get the current CUDA stream");
    }

    mlir::IntegerType const i32 = rewriter.getI32Type();
    mlir::Value const deviceIndexSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        i32,
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                       1));
    mlir::LLVM::CallOp::create(rewriter, loc, *getDeviceIndex, deviceIndexSlot);
    mlir::Value const deviceIndex =
        mlir::LLVM::LoadOp::create(rewriter, loc, i32, deviceIndexSlot);
    mlir::Value const cuda = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i32, DLDeviceType::kDLCUDA);
    mlir::Value const asyncObject =
        mlir::LLVM::CallOp::create(rewriter, loc, *getStream,
                                   mlir::ValueRange{cuda, deviceIndex})
            .getResult();

    mlir::Value const nullPointer =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 0);
    operands.append(2, nullPointer);

    rewriter.replaceOpWithNewOp<mlir::gpu::LaunchFuncOp>(
        op, op.getKernel(), gridSize, blockSize, dynamicSharedMemorySize,
        operands, nullptr, mlir::ValueRange{}, asyncObject, clusterSize);

    return mlir::success();
  }
};

class ConvertTVMFFIToGPUPass final
    : public impl::ConvertTVMFFIToGPUBase<ConvertTVMFFIToGPUPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::TypeConverter typeConverter;
    typeConverter.addConversion(
        [](mlir::Type type) -> mlir::Type { return type; });
    target
        .addIllegalOp<tvm_ffi::KernelLaunchOp, tvm_ffi::NativeKernelLaunchOp>();
    target.addLegalDialect<dlpack::DLPackDialect, mlir::BuiltinDialect,
                           mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                           mlir::cf::ControlFlowDialect,
                           mlir::func::FuncDialect, mlir::gpu::GPUDialect,
                           mlir::scf::SCFDialect, tvm_ffi::TVMFFIDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    populateTVMFFIToGPUConversionPatterns(patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateTVMFFIToGPUConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &typeConverter) {
  patterns.add<ConvertKernelLaunchOp>(typeConverter, patterns.getContext());
}

} // namespace trident::conversion
