//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/ABI/Transforms/LowerTVMFFIABITensorMetadata.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace trident::tvm_ffi::abi {

#define GEN_PASS_DEF_LOWERTVMFFIABITENSORMETADATA
#include "trident/core/Dialect/TVMFFI/ABI/Transforms/Passes.h.inc"

template <typename SourceOp, typename TargetOp>
class LowerIndexedTensorMetadataOp final
    : public mlir::OpRewritePattern<SourceOp> {
public:
  using mlir::OpRewritePattern<SourceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Type const objectType = ObjectType::get(context);
    mlir::Type const tensorType = dlpack::DLTensorType::get(context);
    mlir::Value const object =
        GetOp::create(rewriter, op.getLoc(), objectType, op.getTensor());
    mlir::Value const tensor =
        AsOp::create(rewriter, op.getLoc(), tensorType, object);
    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getResult().getType(), tensor,
                                          op.getIndex());
    return mlir::success();
  }
};

template <typename SourceOp, typename TargetOp>
class LowerTVMFFIABITensorMetadataOp final
    : public mlir::OpRewritePattern<SourceOp> {
public:
  using mlir::OpRewritePattern<SourceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Type const objectType = ObjectType::get(context);
    mlir::Type const tensorType = dlpack::DLTensorType::get(context);
    mlir::Value const object =
        GetOp::create(rewriter, op.getLoc(), objectType, op.getTensor());
    mlir::Value const tensor =
        AsOp::create(rewriter, op.getLoc(), tensorType, object);
    rewriter.replaceOpWithNewOp<TargetOp>(op, op->getResultTypes(), tensor);
    return mlir::success();
  }
};

class LowerTVMFFIABITensorMetadataPass final
    : public impl::LowerTVMFFIABITensorMetadataBase<
          LowerTVMFFIABITensorMetadataPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    populateLowerTVMFFIABITensorMetadataPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateLowerTVMFFIABITensorMetadataPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.add<
      LowerIndexedTensorMetadataOp<TensorSizeOp, dlpack::TensorSizeOp>,
      LowerIndexedTensorMetadataOp<TensorStrideOp, dlpack::TensorStrideOp>,
      LowerTVMFFIABITensorMetadataOp<TensorDTypeOp, dlpack::TensorDTypeOp>,
      LowerTVMFFIABITensorMetadataOp<TensorDeviceOp, dlpack::TensorDeviceOp>,
      LowerTVMFFIABITensorMetadataOp<TensorDimOp, dlpack::TensorDimOp>,
      LowerTVMFFIABITensorMetadataOp<TensorStorageOffsetOp,
                                     dlpack::TensorStorageOffsetOp>>(
      patterns.getContext());
}

} // namespace trident::tvm_ffi::abi
