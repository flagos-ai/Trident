// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/DecomposeTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace trident::tvm_ffi {
#define GEN_PASS_DEF_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"
namespace {

class DecomposeBindOp final : public mlir::OpRewritePattern<MonadBindOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MonadBindOp bind,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(bind);
    mlir::Location const loc = bind.getLoc();
    mlir::Value ready;
    llvm::SmallVector<mlir::Value> unpacked;
    for (mlir::Value const input : bind.getInputs()) {
      MonadUnpackOp unpack = MonadUnpackOp::create(rewriter, loc, input);
      mlir::Value const success = unpack.getSuccess();
      ready = ready ? mlir::arith::AndIOp::create(rewriter, loc, ready, success)
                    : success;
      unpacked.push_back(unpack.getValue());
    }
    // TODO: Eliminate the redundant Any selection and downcasts introduced by
    // unpacking typed Results when expanding this standard bind form.
    mlir::scf::IfOp branch = mlir::scf::IfOp::create(
        rewriter, loc, bind.getResultTypes(), ready, true);
    rewriter.setInsertionPointToStart(branch.thenBlock());
    mlir::Block &body = bind.getBody().front();
    llvm::SmallVector<mlir::Value> payloads;
    for (auto [value, argument] :
         llvm::zip_equal(unpacked, body.getArguments())) {
      if (value.getType() != argument.getType()) {
        value = DowncastOp::create(rewriter, loc, argument.getType(), value);
      }
      payloads.push_back(value);
    }
    // Move the continuation to preserve its lexical captures and effect order.
    MonadYieldOp yielded = mlir::cast<MonadYieldOp>(body.getTerminator());
    rewriter.setInsertionPoint(yielded);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yielded,
                                                    yielded.getValues());
    rewriter.mergeBlocks(&body, branch.thenBlock(), payloads);
    rewriter.setInsertionPointToStart(branch.elseBlock());
    MonadFailOp failed =
        MonadFailOp::create(rewriter, loc, bind.getResultTypes());
    mlir::scf::YieldOp::create(rewriter, loc, failed.getResults());
    rewriter.replaceOp(bind, branch.getResults());
    return mlir::success();
  }
};

class DecomposeTVMFFIPass final
    : public impl::DecomposeTVMFFIBase<DecomposeTVMFFIPass> {
  void runOnOperation() override {
    mlir::RewritePatternSet bindPatterns(&getContext());
    bindPatterns.add<DecomposeBindOp>(&getContext());
    mlir::FrozenRewritePatternSet const frozen(std::move(bindPatterns));
    // Keep canonicalization separate from this structural lowering.
    mlir::GreedyRewriteConfig config;
    config.enableFolding(false).enableConstantCSE(false);
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Disabled);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), frozen, config))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace trident::tvm_ffi
