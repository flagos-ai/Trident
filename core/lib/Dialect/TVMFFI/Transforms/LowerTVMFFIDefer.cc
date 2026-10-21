// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/LowerTVMFFIDefer.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include <cstdint>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <optional>
#include <tuple>
#include <utility>

namespace trident::tvm_ffi {
#define GEN_PASS_DEF_LOWERTVMFFIDEFER
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

namespace {

class LowerTVMFFIDeferPass final
    : public impl::LowerTVMFFIDeferBase<LowerTVMFFIDeferPass> {
  void runOnOperation() override {
    mlir::DominanceInfo dominance;
    llvm::MapVector<mlir::Region *, llvm::SmallVector<DeferOp>> registrations;
    mlir::WalkResult const checked =
        getOperation()->walk([&](DeferOp cleanup) -> mlir::WalkResult {
          mlir::Region *region = cleanup->getParentRegion();
          if (!dominance.hasSSADominance(region)) {
            cleanup.emitOpError(
                "requires an SSACFG region for deferred cleanup");
            return mlir::WalkResult::interrupt();
          }
          registrations[region].push_back(cleanup);
          return mlir::WalkResult::advance();
        });
    if (checked.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    // Plan every site before changing any CFG: both diagnostics and dominance
    // queries must see the original IR.
    llvm::SmallVector<std::tuple<mlir::Operation *, std::optional<uint32_t>,
                                 llvm::SmallVector<DeferOp>>>
        sites;
    for (auto &[region, cleanups] : registrations) {
      for (mlir::Block &block : llvm::make_filter_range(
               *region, [&dominance](mlir::Block &block) -> bool {
                 return dominance.isReachableFromEntry(&block);
               })) {
        llvm::SmallVector<DeferOp> active = llvm::filter_to_vector(
            cleanups, [&block, &dominance](DeferOp cleanup) -> bool {
              return dominance.dominates(cleanup->getBlock(), &block);
            });
        if (active.empty()) {
          continue;
        }
        if (block.empty() ||
            !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
          region->getParentOp()->emitOpError(
              "requires a block terminator for deferred cleanup");
          signalPassFailure();
          return;
        }
        mlir::Operation *terminator = block.getTerminator();
        uint32_t const successorCount = terminator->getNumSuccessors();
        if (successorCount != 0 &&
            !mlir::isa<mlir::BranchOpInterface>(terminator)) {
          region->getParentOp()->emitOpError(
              "unsupported branch for deferred cleanup");
          signalPassFailure();
          return;
        }
        if (successorCount == 0) {
          sites.push_back({terminator, std::nullopt, std::move(active)});
        } else {
          llvm::append_range(
              sites,
              llvm::map_range(
                  llvm::seq(successorCount),
                  [&](uint32_t index)
                      -> std::tuple<mlir::Operation *, std::optional<uint32_t>,
                                    llvm::SmallVector<DeferOp>> {
                    mlir::Block *destination = terminator->getSuccessor(index);
                    llvm::SmallVector<DeferOp> exiting = llvm::filter_to_vector(
                        active,
                        [&dominance, destination](DeferOp cleanup) -> bool {
                          mlir::Block *registration = cleanup->getBlock();
                          // Re-entry ends the previous registration before the
                          // next iteration.
                          return destination == registration ||
                                 !dominance.dominates(registration,
                                                      destination);
                        });
                    return {terminator, index, std::move(exiting)};
                  }));
        }
      }
    }
    mlir::OpBuilder builder(&getContext());
    for (const auto &[terminator, successorIndex, cleanups] : sites) {
      builder.setInsertionPoint(terminator);
      if (successorIndex) {
        // Split by index: repeated destinations can carry different operands.
        mlir::Block *destination = terminator->getSuccessor(*successorIndex);
        mlir::Block *edge = builder.createBlock(
            destination->getParent(), destination->getIterator(),
            destination->getArgumentTypes(),
            llvm::SmallVector<mlir::Location>(destination->getNumArguments(),
                                              terminator->getLoc()));
        terminator->setSuccessor(edge, *successorIndex);
        mlir::cf::BranchOp const branch = mlir::cf::BranchOp::create(
            builder, terminator->getLoc(), destination, edge->getArguments());
        builder.setInsertionPoint(branch);
      }
      for (DeferOp cleanup : cleanups) {
        mlir::IRMapping mapping;
        for (mlir::Operation &operation :
             cleanup.getEntryBlock()->without_terminator()) {
          builder.clone(operation, mapping);
        }
      }
    }
    // Unreachable registrations cannot execute and need no cleanup copies.
    for (auto &[region, cleanups] : registrations) {
      for (DeferOp cleanup : cleanups) {
        cleanup.erase();
      }
    }
  }
};

} // namespace
} // namespace trident::tvm_ffi
