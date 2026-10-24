//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/Transforms/LowerToPack.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace trident::tvm_ffi {

#define GEN_PASS_DEF_LOWERTOPACK
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

class ConvertFuncOp final : public mlir::OpConversionPattern<FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FuncOp function, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type const any = CAnyType::get(function.getContext());
    mlir::Type const pack = PackArgumentsType::get(function.getContext());
    PackFuncOp packed =
        PackFuncOp::create(rewriter, function.getLoc(), function.getSymName(),
                           rewriter.getFunctionType({pack}, {any}));
    packed.setSymVisibilityAttr(function.getSymVisibilityAttr());
    if (function.getBody().empty()) {
      rewriter.eraseBlock(packed.getEntryBlock());
      rewriter.eraseOp(function);
      return mlir::success();
    }
    for (mlir::Block &block : function.getBody()) {
      if (ReturnOp returned = mlir::dyn_cast<ReturnOp>(block.getTerminator())) {
        // Preserve the typed payload contract on successful returns.
        mlir::Type const payloadType = function.getResultTypes().front();
        mlir::Location const loc = returned.getLoc();
        mlir::Type const any = CAnyType::get(returned.getContext());
        rewriter.setInsertionPoint(returned);
        mlir::Value const ready = GetOp::create(
            rewriter, loc, rewriter.getI1Type(), returned.getSuccess());
        mlir::scf::IfOp branch = mlir::scf::IfOp::create(
            rewriter, loc, mlir::TypeRange{any}, ready, true);
        rewriter.setInsertionPointToStart(branch.thenBlock());
        mlir::Value const checked = rewriter.createOrFold<DowncastOp>(
            loc, payloadType, returned.getValue());
        mlir::Value const value =
            rewriter.createOrFold<UpcastOp>(loc, any, checked);
        mlir::scf::YieldOp::create(rewriter, loc, value);
        rewriter.setInsertionPointToStart(branch.elseBlock());
        mlir::Value const none = ConstantNoneOp::create(rewriter, loc);
        mlir::Value const failed = UpcastOp::create(rewriter, loc, any, none);
        mlir::scf::YieldOp::create(rewriter, loc, failed);
        rewriter.modifyOpInPlace(returned, [&] {
          returned.getValueMutable().assign(branch.getResult(0));
        });
      }
    }
    // Check arity before reading slots and tags before entering the typed CFG.
    mlir::Location const loc = function.getLoc();
    mlir::Block *entry = packed.getEntryBlock();
    mlir::Block *unpack = rewriter.createBlock(&packed.getBody());
    mlir::Block *accepted = rewriter.createBlock(&packed.getBody());
    mlir::Block *rejected = rewriter.createBlock(&packed.getBody());
    rewriter.setInsertionPointToEnd(entry);
    mlir::Value const arguments = entry->getArgument(0);
    mlir::Value const count = PackSizeOp::create(rewriter, loc, arguments);
    mlir::Value const expected = mlir::arith::ConstantIntOp::create(
        rewriter, loc, function.getNumArguments(), 64);
    mlir::Value const arity = mlir::arith::CmpIOp::create(
        rewriter, loc, mlir::arith::CmpIPredicate::eq, count, expected);
    mlir::cf::CondBranchOp::create(rewriter, loc, arity, unpack, rejected);

    rewriter.setInsertionPointToEnd(unpack);
    llvm::SmallVector<mlir::Value> values = llvm::map_to_vector(
        llvm::seq(function.getNumArguments()),
        [&rewriter, &loc, &any, &arguments](uint32_t index) -> mlir::Value {
          return PackGetOp::create(
              rewriter, loc, any, arguments,
              mlir::arith::ConstantIntOp::create(
                  rewriter, loc, static_cast<int64_t>(index), 64));
        });
    mlir::Value const matches = llvm::accumulate(
        llvm::zip_equal(values, function.getArgumentTypes()),
        mlir::arith::ConstantIntOp::create(rewriter, loc, 1, 1).getResult(),
        [&rewriter, loc](mlir::Value matches, auto argument) -> mlir::Value {
          auto [value, argumentType] = argument;
          return mlir::isa<CAnyType>(argumentType)
                     ? matches
                     : mlir::arith::AndIOp::create(
                           rewriter, loc, matches,
                           IsaOp::create(rewriter, loc, value, argumentType));
        });
    mlir::cf::CondBranchOp::create(rewriter, loc, matches, accepted, rejected);
    rewriter.setInsertionPointToEnd(accepted);
    for (auto [value, argumentType] :
         llvm::zip_equal(values, function.getArgumentTypes())) {
      value = rewriter.createOrFold<DowncastOp>(loc, argumentType, value);
    }
    mlir::cf::BranchOp::create(rewriter, loc, function.getEntryBlock(), values);
    rewriter.setInsertionPointToEnd(rejected);
    // Ownership decomposition has already run. Transfer the Base exception's
    // newly acquired reference directly to the packed caller.
    mlir::Value const exception = ExceptionOp::create(
        rewriter, loc, CExceptionType::get(function.getContext()),
        "GuardMatch");
    mlir::Value const value = UpcastOp::create(rewriter, loc, any, exception);
    mlir::Value const success = ConstantBoolOp::create(rewriter, loc, true);
    ReturnOp::create(rewriter, loc, success, value);
    rewriter.inlineRegionBefore(function.getBody(), packed.getBody(),
                                packed.getBody().end());
    rewriter.eraseOp(function);
    return mlir::success();
  }
};

class LowerToPackPass final : public impl::LowerToPackBase<LowerToPackPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    populateLowerToPackConversionPatterns(target, patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateLowerToPackConversionPatterns(mlir::ConversionTarget &target,
                                           mlir::RewritePatternSet &patterns) {
  target.addIllegalOp<FuncOp>();
  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *) -> bool { return true; });
  patterns.add<ConvertFuncOp>(patterns.getContext());
}

} // namespace trident::tvm_ffi
