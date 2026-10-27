//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace trident::tvm_ffi {

namespace {

class FoldTupleConstructUnconstruct final
    : public mlir::OpRewritePattern<TupleUnconstructOp> {
public:
  using mlir::OpRewritePattern<TupleUnconstructOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TupleUnconstructOp op,
                  mlir::PatternRewriter &rewriter) const override {
    TupleConstructOp construct =
        op.getContainer().getDefiningOp<TupleConstructOp>();
    if (!construct || !construct.getResult().hasOneUse()) {
      return rewriter.notifyMatchFailure(
          op, "tuple must be produced by a single-use tuple.construct");
    }
    rewriter.replaceOp(op, construct.getElements());
    rewriter.eraseOp(construct);
    return mlir::success();
  }
};

} // namespace

void TupleUnconstructOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<FoldTupleConstructUnconstruct>(context);
}

} // namespace trident::tvm_ffi
