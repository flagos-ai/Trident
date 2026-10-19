// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/MaterializeOwnershipMark.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <utility>

namespace trident::tvm_ffi {

namespace {

constexpr llvm::StringLiteral cleanupAttributeName = "tvm_ffi.cleanup";
constexpr llvm::StringLiteral retainAttributeName = "tvm_ffi.retain";

bool needsReference(mlir::Type type) {
  if (ResultType const result = mlir::dyn_cast<ResultType>(type)) {
    type = result.getPayloadType();
  }
  return type && (mlir::isa<ObjectType>(type) ||
                  type.hasTrait<mlir::TypeTrait::HasObject>() ||
                  type.hasTrait<mlir::TypeTrait::Contains>());
}

template <typename Op>
void adjustReference(mlir::OpBuilder &builder, mlir::Value value) {
  if (needsReference(value.getType())) {
    mlir::Location const loc = value.getLoc();
    mlir::OpBuilder::InsertionGuard const guard(builder);
    if (mlir::isa<ResultType>(value.getType())) {
      MonadUnpackOp unpack = MonadUnpackOp::create(builder, loc, value);
      mlir::scf::IfOp branch = mlir::scf::IfOp::create(
          builder, loc, mlir::TypeRange{}, unpack.getSuccess(), false);
      builder.setInsertionPoint(branch.thenBlock()->getTerminator());
      value = unpack.getValue();
    }
    Op::create(builder, loc, value);
  }
}

// Frontend pure borrows its inputs. Materialize their independent references
// after semantic containers have become CArray, once per payload occurrence.
class RetainPayloadsPattern final : public mlir::OpRewritePattern<MonadPureOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MonadPureOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasAttr("tvm_ffi.retain_payloads")) {
      return mlir::failure();
    }
    rewriter.setInsertionPoint(op);
    for (mlir::Value const value : op.getValues()) {
      adjustReference<ObjectIncRefOp>(rewriter, value);
    }
    rewriter.modifyOpInPlace(
        op, [&] { op->removeAttr("tvm_ffi.retain_payloads"); });
    return mlir::success();
  }
};

class RetainExportsPattern final : public mlir::RewritePattern {
public:
  explicit RetainExportsPattern(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *operation,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::Region *region = operation->getParentRegion();
    if (!region || !region->hasOneBlock() ||
        !mlir::isa<mlir::func::ReturnOp, ReturnOp, MonadYieldOp,
                   mlir::scf::YieldOp>(operation) ||
        !operation->hasAttr(retainAttributeName)) {
      return mlir::failure();
    }
    rewriter.setInsertionPoint(operation);
    for (mlir::Value const value : operation->getOperands()) {
      adjustReference<ObjectIncRefOp>(rewriter, value);
    }
    rewriter.modifyOpInPlace(
        operation, [&] { operation->removeAttr(retainAttributeName); });
    return mlir::success();
  }
};

class CleanupBlockPattern final : public mlir::RewritePattern {
public:
  explicit CleanupBlockPattern(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *operation,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::Region *region = operation->getParentRegion();
    if (!region || !region->hasOneBlock() ||
        !operation->hasTrait<mlir::OpTrait::IsTerminator>()) {
      return mlir::failure();
    }
    // Retains from the first stage already precede this insertion point.
    rewriter.setInsertionPoint(operation);
    bool changed = false;
    for (mlir::Operation &local :
         llvm::make_early_inc_range(llvm::reverse(*operation->getBlock()))) {
      if (local.hasAttr(cleanupAttributeName)) {
        for (mlir::Value const value : llvm::reverse(local.getResults())) {
          adjustReference<ObjectDecRefOp>(rewriter, value);
        }
        rewriter.modifyOpInPlace(
            &local, [&] { local.removeAttr(cleanupAttributeName); });
        changed = true;
      }
    }
    return mlir::success(changed);
  }
};

} // namespace

#define GEN_PASS_DEF_MATERIALIZETVMFFICLEANUP
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

class MaterializeTVMFFICleanupPass final
    : public impl::MaterializeTVMFFICleanupBase<MaterializeTVMFFICleanupPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet payloadPatterns(&getContext());
    payloadPatterns.add<RetainPayloadsPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(payloadPatterns));

    mlir::RewritePatternSet retainPatterns(&getContext());
    retainPatterns.add<RetainExportsPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(retainPatterns));

    mlir::RewritePatternSet cleanupPatterns(&getContext());
    cleanupPatterns.add<CleanupBlockPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(cleanupPatterns));
  }
};

} // namespace trident::tvm_ffi
