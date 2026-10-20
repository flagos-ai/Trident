// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/MaterializeOwnershipMark.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
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

bool needsReference(mlir::Type type) {
  return type && (mlir::isa<ObjectType>(type) ||
                  type.hasTrait<mlir::TypeTrait::HasObject>() ||
                  type.hasTrait<mlir::TypeTrait::Contains>());
}

template <typename Op>
void adjustReference(mlir::OpBuilder &builder, mlir::Value value) {
  if (needsReference(value.getType())) {
    mlir::Location const loc = value.getLoc();
    mlir::OpBuilder::InsertionGuard const guard(builder);
    Op::create(builder, loc, value);
  }
}

// Materialize independent references for borrowed pure payloads and exports
// after semantic containers have become CArray, once per operand occurrence.
class RetainPattern final : public mlir::RewritePattern {
public:
  explicit RetainPattern(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *operation,
                  mlir::PatternRewriter &rewriter) const final {
    if (!operation->hasAttr(kRetainAttributeName)) {
      return mlir::failure();
    }
    mlir::Region *region = operation->getParentRegion();
    if (!region || !region->hasOneBlock() ||
        !mlir::isa<mlir::func::ReturnOp, ReturnOp, mlir::scf::YieldOp>(
            operation)) {
      return mlir::failure();
    }
    rewriter.setInsertionPoint(operation);
    for (mlir::Value const value : operation->getOperands()) {
      adjustReference<ObjectIncRefOp>(rewriter, value);
    }
    rewriter.modifyOpInPlace(
        operation, [&] { operation->removeAttr(kRetainAttributeName); });
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
    for (mlir::Operation &local : llvm::reverse(*operation->getBlock())) {
      if (local.hasAttr(kCleanupAttributeName)) {
        for (mlir::Value const value : llvm::reverse(local.getResults())) {
          adjustReference<ObjectDecRefOp>(rewriter, value);
        }
        rewriter.modifyOpInPlace(
            &local, [&] { local.removeAttr(kCleanupAttributeName); });
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
    mlir::RewritePatternSet retainPatterns(&getContext());
    retainPatterns.add<RetainPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(retainPatterns));

    mlir::RewritePatternSet cleanupPatterns(&getContext());
    cleanupPatterns.add<CleanupBlockPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(cleanupPatterns));
  }
};

} // namespace trident::tvm_ffi
