// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
#include "trident/core/Dialect/TVMFFI/API/Transforms/DirectAllocations.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace trident::tvm_ffi::api {
#define GEN_PASS_DEF_DIRECTALLOCATIONS
#define GEN_PASS_DEF_FLATTENDIRECTALLOCATIONS
#define GEN_PASS_DEF_UNBOXSCFIFTUPLES
#include "trident/core/Dialect/TVMFFI/API/Transforms/Passes.h.inc"

static int classifyDirectAllocation(llvm::StringRef name, size_t argc) {
  if (name == "trident.aten.empty.memory_format" && argc == 6)
    return 0;
  if (name == "trident.aten.empty_strided" && argc == 6)
    return 1;
  if (name == "trident.aten.empty_like" && argc == 6)
    return 2;
  if (name == "trident.aten.zeros" && argc == 5)
    return 3;
  if (name == "trident.aten.zeros_like" && argc == 6)
    return 4;
  if ((name == "trident.aten.view" || name == "trident.aten._unsafe_view") &&
      argc == 2)
    return 5;
  return -1;
}

class DirectAllocationsPass final
    : public impl::DirectAllocationsBase<DirectAllocationsPass> {
  void runOnOperation() final {
    llvm::SmallVector<mlir::Operation *> calls;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<GlobalCallOp, FunctionCallOp>(op))
        calls.push_back(op);
    });
    mlir::IRRewriter rw(&getContext());
    for (auto *callOp : calls) {
      llvm::StringRef name;
      mlir::ValueRange arguments;
      mlir::Value callResult;
      FunctionGetGlobalOp global;
      if (auto globalCall = mlir::dyn_cast<GlobalCallOp>(callOp)) {
        name = globalCall.getName();
        arguments = globalCall.getArguments();
        callResult = globalCall.getResult();
      } else {
        auto fnCall = mlir::cast<FunctionCallOp>(callOp);
        global = fnCall.getCallee().getDefiningOp<FunctionGetGlobalOp>();
        if (!global)
          continue;
        name = global.getName();
        arguments = fnCall.getArguments();
        callResult = fnCall.getResult();
      }
      int kind = classifyDirectAllocation(name, arguments.size());
      if (kind < 0)
        continue;
      mlir::Value value = callResult;
      llvm::SmallVector<mlir::Operation *> casts;
      while (value.hasOneUse()) {
        auto cast = mlir::dyn_cast<UpcastOp>(*value.getUsers().begin());
        if (!cast)
          break;
        casts.push_back(cast);
        value = cast.getResult();
      }
      if (!value.hasOneUse())
        continue;
      auto pure =
          mlir::dyn_cast<tvm_ffi::monad::PureOp>(*value.getUsers().begin());
      // Only Result producers are rewritten: this preserves failure
      // propagation.
      if (!pure)
        continue;
      rw.setInsertionPoint(callOp);
      auto direct = tvm_ffi::monad::DirectAllocationOp::create(
          rw, callOp->getLoc(), pure.getOutput().getType(),
          rw.getI32IntegerAttr(kind), arguments);
      rw.replaceOp(pure, direct.getResult());
      for (auto *cast : llvm::reverse(casts))
        rw.eraseOp(cast);
      rw.eraseOp(callOp);
      if (global && global.getResult().use_empty())
        rw.eraseOp(global);
    }
  }
};

class FlattenDirectAllocationsPass final
    : public impl::FlattenDirectAllocationsBase<FlattenDirectAllocationsPass> {
  void runOnOperation() final {
    getOperation().walk([&](DirectAllocationOp op) {
      int kind = op.getKind();
      if (kind == 2 || kind == 4 || op.getRankAttr().getInt() != -1)
        return;
      auto args = op.getArguments();
      if (kind == 5) {
        auto sizes = args[1].getDefiningOp<ArrayCreateOp>();
        if (!sizes)
          return;
        mlir::IRRewriter rw(&getContext());
        rw.setInsertionPoint(op);
        llvm::SmallVector<mlir::Value> flat;
        flat.push_back(args[0]);
        llvm::append_range(flat, sizes.getElements());
        auto direct = DirectAllocationOp::create(
            rw, op.getLoc(), op->getResultTypes(), op.getKindAttr(),
            rw.getI64IntegerAttr(sizes.getElements().size()), flat);
        rw.replaceOp(op, direct->getResults());
        if (sizes.getResult().use_empty())
          rw.eraseOp(sizes);
        return;
      }
      auto sizes = args[0].getDefiningOp<ArrayCreateOp>();
      if (!sizes)
        return;
      ArrayCreateOp strides;
      if (kind == 1) {
        strides = args[1].getDefiningOp<ArrayCreateOp>();
        if (!strides ||
            strides.getElements().size() != sizes.getElements().size())
          return;
      }
      mlir::IRRewriter rw(&getContext());
      rw.setInsertionPoint(op);
      llvm::SmallVector<mlir::Value> flat(sizes.getElements());
      if (strides)
        llvm::append_range(flat, strides.getElements());
      llvm::append_range(flat, args.drop_front(strides ? 2 : 1));
      auto direct = DirectAllocationOp::create(
          rw, op.getLoc(), op->getResultTypes(), op.getKindAttr(),
          rw.getI64IntegerAttr(sizes.getElements().size()), flat);
      rw.replaceOp(op, direct->getResults());
      if (sizes.getResult().use_empty())
        rw.eraseOp(sizes);
      if (strides && strides != sizes && strides.getResult().use_empty())
        rw.eraseOp(strides);
    });
  }
};

namespace {

// Unbox TupleType results across scf.if boundaries so Monad Results and
// KernelLaunch outcomes stay in SSA registers instead of heap-allocating
// ffi.Array objects and calling ffi.ArrayGetItem on every continuation.
struct UnboxScfIfTupleResults final
    : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getNumResults() == 0 || op.getElseRegion().empty())
      return mlir::failure();
    auto thenYield =
        mlir::dyn_cast<mlir::scf::YieldOp>(op.thenBlock()->getTerminator());
    auto elseYield =
        mlir::dyn_cast<mlir::scf::YieldOp>(op.elseBlock()->getTerminator());
    if (!thenYield || !elseYield)
      return mlir::failure();

    llvm::SmallVector<bool> unbox(op.getNumResults(), false);
    bool anyUnbox = false;
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      auto tupleTy = mlir::dyn_cast<TupleType>(op.getResult(i).getType());
      if (!tupleTy)
        continue;
      auto thenCtor = thenYield.getOperand(i).getDefiningOp<TupleConstructOp>();
      auto elseCtor = elseYield.getOperand(i).getDefiningOp<TupleConstructOp>();
      bool allUsersUnconstruct =
          !op.getResult(i).use_empty() &&
          llvm::all_of(op.getResult(i).getUsers(), [](mlir::Operation *user) {
            return mlir::isa<TupleUnconstructOp>(user);
          });
      if ((thenCtor && elseCtor) ||
          ((thenCtor || elseCtor) && allUsersUnconstruct)) {
        unbox[i] = true;
        anyUnbox = true;
      }
    }
    if (!anyUnbox)
      return mlir::failure();

    // Never leave an scf.if with all-ABI result types if any element is an
    // opaque ABI object: ConvertTVMFFIAPIToABI treats all-ABI scf.if/scf.yield
    // as already legal and would skip inserting boundary IncRef/DecRef ops.
    bool hasOpaqueAbiResult = false;
    bool hasApiResult = false;
    auto inspectType = [&](mlir::Type t) {
      if (mlir::isa<TVMFFIAPIDialect>(t.getDialect()))
        hasApiResult = true;
      else if (mlir::isa<TVMFFICOpaqueABIType>(t))
        hasOpaqueAbiResult = true;
    };
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      if (!unbox[i]) {
        inspectType(op.getResult(i).getType());
      } else {
        for (mlir::Type elemTy :
             mlir::cast<TupleType>(op.getResult(i).getType()).getTypes())
          inspectType(elemTy);
      }
    }
    if (hasOpaqueAbiResult && !hasApiResult)
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Type> newResultTypes;
    llvm::SmallVector<mlir::Value> newThenArgs;
    llvm::SmallVector<mlir::Value> newElseArgs;

    auto appendBranchArgs = [&](mlir::scf::YieldOp yieldOp, unsigned index,
                                TupleType tupleTy,
                                llvm::SmallVectorImpl<mlir::Value> &out) {
      mlir::Value val = yieldOp.getOperand(index);
      if (auto ctor = val.getDefiningOp<TupleConstructOp>()) {
        llvm::append_range(out, ctor.getElements());
      } else {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(yieldOp);
        llvm::SmallVector<mlir::Type> elemTypes(tupleTy.getTypes().begin(),
                                                tupleTy.getTypes().end());
        auto un = TupleUnconstructOp::create(rewriter, yieldOp.getLoc(),
                                             elemTypes, val);
        llvm::append_range(out, un.getElements());
      }
    };

    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      if (!unbox[i]) {
        newResultTypes.push_back(op.getResult(i).getType());
        newThenArgs.push_back(thenYield.getOperand(i));
        newElseArgs.push_back(elseYield.getOperand(i));
      } else {
        auto tupleTy = mlir::cast<TupleType>(op.getResult(i).getType());
        llvm::append_range(newResultTypes, tupleTy.getTypes());
        appendBranchArgs(thenYield, i, tupleTy, newThenArgs);
        appendBranchArgs(elseYield, i, tupleTy, newElseArgs);
      }
    }

    auto newIf = mlir::scf::IfOp::create(rewriter, loc, newResultTypes,
                                         op.getCondition(), false, false);
    newIf->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(),
                                newIf.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(),
                                newIf.getElseRegion().end());

    rewriter.modifyOpInPlace(thenYield,
                             [&]() { thenYield->setOperands(newThenArgs); });
    rewriter.modifyOpInPlace(elseYield,
                             [&]() { elseYield->setOperands(newElseArgs); });

    rewriter.setInsertionPointAfter(newIf);
    llvm::SmallVector<mlir::Value> replacements;
    replacements.reserve(op.getNumResults());
    size_t offset = 0;
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
      if (!unbox[i]) {
        replacements.push_back(newIf.getResult(offset++));
      } else {
        auto tupleTy = mlir::cast<TupleType>(op.getResult(i).getType());
        size_t count = tupleTy.getTypes().size();
        auto reconstructed = TupleConstructOp::create(
            rewriter, loc, tupleTy, newIf.getResults().slice(offset, count));
        replacements.push_back(reconstructed.getResult());
        offset += count;
      }
    }
    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }
};

// Lower primitive scalar equality ops directly to native arith comparisons so
// constexpr checks and scalar guards avoid calling ffi.StructuralEqual and can
// constant-fold before LLVM conversion.
template <typename EqOpTy>
struct SimplifyScalarEq final : public mlir::OpRewritePattern<EqOpTy> {
  using mlir::OpRewritePattern<EqOpTy>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EqOpTy op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();
    mlir::Type lhsTy = lhs.getType();
    mlir::Type rhsTy = rhs.getType();

    if (mlir::isa<abi::NoneType>(lhsTy) && mlir::isa<abi::NoneType>(rhsTy)) {
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(op, 1, 1);
      return mlir::success();
    }
    if (mlir::isa<abi::BoolType>(lhsTy) && mlir::isa<abi::BoolType>(rhsTy)) {
      mlir::Value a =
          abi::GetOp::create(rewriter, loc, rewriter.getI1Type(), lhs);
      mlir::Value b =
          abi::GetOp::create(rewriter, loc, rewriter.getI1Type(), rhs);
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
          op, mlir::arith::CmpIPredicate::eq, a, b);
      return mlir::success();
    }
    if (mlir::isa<abi::IntType>(lhsTy) && mlir::isa<abi::IntType>(rhsTy)) {
      mlir::Value a =
          abi::GetOp::create(rewriter, loc, rewriter.getI64Type(), lhs);
      mlir::Value b =
          abi::GetOp::create(rewriter, loc, rewriter.getI64Type(), rhs);
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
          op, mlir::arith::CmpIPredicate::eq, a, b);
      return mlir::success();
    }
    if (mlir::isa<abi::FloatType>(lhsTy) && mlir::isa<abi::FloatType>(rhsTy)) {
      mlir::Value a =
          abi::GetOp::create(rewriter, loc, rewriter.getF64Type(), lhs);
      mlir::Value b =
          abi::GetOp::create(rewriter, loc, rewriter.getF64Type(), rhs);
      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
          op, mlir::arith::CmpFPredicate::OEQ, a, b);
      return mlir::success();
    }
    if (mlir::isa<api::AnyType, abi::AnyType>(lhsTy) &&
        mlir::isa<abi::NoneType>(rhsTy)) {
      if constexpr (std::is_same_v<EqOpTy, api::EqOp>) {
        rewriter.replaceOpWithNewOp<api::IsaOp>(
            op, lhs, abi::NoneType::get(rewriter.getContext()));
      } else {
        rewriter.replaceOpWithNewOp<abi::IsaOp>(
            op, lhs, abi::NoneType::get(rewriter.getContext()));
      }
      return mlir::success();
    }
    if (mlir::isa<abi::NoneType>(lhsTy) &&
        mlir::isa<api::AnyType, abi::AnyType>(rhsTy)) {
      if constexpr (std::is_same_v<EqOpTy, api::EqOp>) {
        rewriter.replaceOpWithNewOp<api::IsaOp>(
            op, rhs, abi::NoneType::get(rewriter.getContext()));
      } else {
        rewriter.replaceOpWithNewOp<abi::IsaOp>(
            op, rhs, abi::NoneType::get(rewriter.getContext()));
      }
      return mlir::success();
    }
    return mlir::failure();
  }
};

} // namespace

class UnboxScfIfTuplesPass final
    : public impl::UnboxScfIfTuplesBase<UnboxScfIfTuplesPass> {
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<UnboxScfIfTupleResults, SimplifyScalarEq<api::EqOp>,
                 SimplifyScalarEq<abi::EqOp>>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace trident::tvm_ffi::api
