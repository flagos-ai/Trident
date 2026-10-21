// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/LowerTVMFFIDefer.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

namespace trident::tvm_ffi {
#define GEN_PASS_DEF_LOWERTVMFFIDEFER
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

namespace {

class LowerTVMFFIDeferPass final
    : public impl::LowerTVMFFIDeferBase<LowerTVMFFIDeferPass> {
  void runOnOperation() override {
    llvm::SmallVector<DeferOp> cleanups;
    mlir::WalkResult const checked =
        getOperation()->walk([&](DeferOp cleanup) -> mlir::WalkResult {
          mlir::Region *region = cleanup->getParentRegion();
          if (!mlir::isa<FuncOp, mlir::func::FuncOp, mlir::scf::IfOp>(
                  region->getParentOp())) {
            cleanup.emitOpError(
                "unsupported enclosing operation for deferred cleanup");
            return mlir::WalkResult::interrupt();
          }
          if (!region->hasOneBlock()) {
            cleanup.emitOpError("requires a single-block enclosing region");
            return mlir::WalkResult::interrupt();
          }
          if (!mlir::isa<ReturnOp, mlir::func::ReturnOp, mlir::scf::YieldOp>(
                  cleanup->getBlock()->getTerminator())) {
            cleanup.emitOpError("unsupported exit for deferred cleanup");
            return mlir::WalkResult::interrupt();
          }
          cleanups.push_back(cleanup);
          return mlir::WalkResult::advance();
        });
    if (checked.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Later registrations run first; operations within each body keep their
    // order.
    for (DeferOp cleanup : llvm::reverse(cleanups)) {
      mlir::Operation *terminator = cleanup->getBlock()->getTerminator();
      for (mlir::Operation &operation : llvm::make_early_inc_range(
               cleanup.getBody().front().without_terminator())) {
        operation.moveBefore(terminator);
      }
      cleanup.erase();
    }
  }
};

} // namespace
} // namespace trident::tvm_ffi
