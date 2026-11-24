// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Arith/Transforms/LowerTVMFFIArith.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/Arith/IR/TVMFFIOps.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace trident::tvm_ffi::arith {

#define GEN_PASS_DEF_LOWERTVMFFIARITH
#include "trident/core/Dialect/TVMFFI/Arith/Transforms/Passes.h.inc"

namespace {

// Mixed operand types require promotion before selecting the native operation.
// Keep this dynamic part in one C++ base instead of duplicating DRR/PDLL native
// callbacks for each arithmetic operation and operand combination.
template <typename SourceOp>
class ArithOpLowering : public mlir::OpRewritePattern<SourceOp> {
public:
  using mlir::OpRewritePattern<SourceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Type> const promotedType =
        SourceOp::getArithType(op->getOperandTypes());
    if (mlir::failed(promotedType)) {
      return mlir::failure();
    }
    mlir::Type const arithType = *promotedType;
    mlir::Type const nativeType =
        mlir::cast<TVMFFINativeTypeInterface>(arithType).getNativeType();
    llvm::SmallVector<mlir::Value> const operands = llvm::map_to_vector(
        op->getOperands(), [&](mlir::Value const operand) -> mlir::Value {
          mlir::Value const value =
              abi::GetOp::create(rewriter, op.getLoc(), operand);
          return llvm::TypeSwitch<mlir::Type, mlir::Value>(operand.getType())
              .template Case<abi::BoolType>([&](abi::BoolType) -> mlir::Value {
                return llvm::TypeSwitch<mlir::Type, mlir::Value>(arithType)
                    .template Case<abi::FloatType>(
                        [&](abi::FloatType) -> mlir::Value {
                          return mlir::arith::UIToFPOp::create(
                              rewriter, op.getLoc(), nativeType, value);
                        })
                    .template Case<abi::IntType>(
                        [&](abi::IntType) -> mlir::Value {
                          return mlir::arith::ExtUIOp::create(
                              rewriter, op.getLoc(), nativeType, value);
                        })
                    .Default(value);
              })
              .template Case<abi::IntType>([&](abi::IntType) -> mlir::Value {
                return llvm::TypeSwitch<mlir::Type, mlir::Value>(arithType)
                    .template Case<abi::FloatType>(
                        [&](abi::FloatType) -> mlir::Value {
                          return mlir::arith::SIToFPOp::create(
                              rewriter, op.getLoc(), nativeType, value);
                        })
                    .Default(value);
              })
              .Default(value);
        });
    mlir::Value const result =
        buildNativeOperation(op.getLoc(), operands, rewriter);
    rewriter.replaceOpWithNewOp<abi::ToOp>(op, result);
    return mlir::success();
  }

private:
  virtual mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const = 0;
};

template <typename SourceOp, typename IntegerOp, typename FloatOp = void>
class BinaryArithOpLowering final : public ArithOpLowering<SourceOp> {
public:
  using ArithOpLowering<SourceOp>::ArithOpLowering;

private:
  mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const final {
    if (mlir::isa<mlir::FloatType>(operands[0].getType())) {
      return FloatOp::create(rewriter, loc, operands[0], operands[1]);
    } else {
      return IntegerOp::create(rewriter, loc, operands[0], operands[1]);
    }
  }
};

template <typename SourceOp, typename IntegerOp>
class BinaryArithOpLowering<SourceOp, IntegerOp, void> final
    : public ArithOpLowering<SourceOp> {
public:
  using ArithOpLowering<SourceOp>::ArithOpLowering;

private:
  mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const final {
    return IntegerOp::create(rewriter, loc, operands[0], operands[1]);
  }
};

template <typename SourceOp, typename FloatOp>
class BinaryArithOpLowering<SourceOp, void, FloatOp> final
    : public ArithOpLowering<SourceOp> {
public:
  using ArithOpLowering<SourceOp>::ArithOpLowering;

private:
  mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const final {
    return FloatOp::create(rewriter, loc, operands[0], operands[1]);
  }
};

template <typename SourceOp, mlir::arith::CmpIPredicate IntegerPredicate,
          mlir::arith::CmpFPredicate FloatPredicate>
class ComparisonOpLowering final : public ArithOpLowering<SourceOp> {
public:
  using ArithOpLowering<SourceOp>::ArithOpLowering;

private:
  mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const final {
    if (mlir::isa<mlir::FloatType>(operands[0].getType())) {
      return mlir::arith::CmpFOp::create(rewriter, loc, FloatPredicate,
                                         operands[0], operands[1]);
    } else {
      return mlir::arith::CmpIOp::create(rewriter, loc, IntegerPredicate,
                                         operands[0], operands[1]);
    }
  }
};

class ModOpLowering final : public ArithOpLowering<ModOp> {
public:
  using ArithOpLowering::ArithOpLowering;

private:
  mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const final {
    mlir::Value const quotient = mlir::arith::FloorDivSIOp::create(
        rewriter, loc, operands[0], operands[1]);
    mlir::Value const product =
        mlir::arith::MulIOp::create(rewriter, loc, quotient, operands[1]);
    return mlir::arith::SubIOp::create(rewriter, loc, operands[0], product);
  }
};

class NegOpLowering final : public ArithOpLowering<NegOp> {
public:
  using ArithOpLowering::ArithOpLowering;

private:
  mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const final {
    if (mlir::isa<mlir::FloatType>(operands[0].getType())) {
      return mlir::arith::NegFOp::create(rewriter, loc, operands[0]);
    } else {
      mlir::Value const zero =
          mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      return mlir::arith::SubIOp::create(rewriter, loc, zero, operands[0]);
    }
  }
};

class NotOpLowering final : public ArithOpLowering<NotOp> {
public:
  using ArithOpLowering::ArithOpLowering;

private:
  mlir::Value
  buildNativeOperation(mlir::Location loc, mlir::ValueRange operands,
                       mlir::PatternRewriter &rewriter) const final {
    if (mlir::isa<mlir::FloatType>(operands[0].getType())) {
      mlir::Value const zero = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getF64FloatAttr(0.0));
      // Ordered equality makes NaN truthy and both floating zeros falsy.
      return mlir::arith::CmpFOp::create(
          rewriter, loc, mlir::arith::CmpFPredicate::OEQ, operands[0], zero);
    } else {
      mlir::Value const zero =
          mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      return mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::eq, operands[0], zero);
    }
  }
};

class LowerTVMFFIArithPass final
    : public impl::LowerTVMFFIArithBase<LowerTVMFFIArithPass> {
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    populateLowerTVMFFIArithPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateLowerTVMFFIArithPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<
      BinaryArithOpLowering<AddOp, mlir::arith::AddIOp, mlir::arith::AddFOp>,
      BinaryArithOpLowering<AndOp, mlir::arith::AndIOp>,
      BinaryArithOpLowering<FloorDivOp, mlir::arith::FloorDivSIOp>,
      BinaryArithOpLowering<LShiftOp, mlir::arith::ShLIOp>,
      BinaryArithOpLowering<MaxOp, mlir::arith::MaxSIOp,
                            mlir::arith::MaximumFOp>,
      BinaryArithOpLowering<MinOp, mlir::arith::MinSIOp,
                            mlir::arith::MinimumFOp>,
      BinaryArithOpLowering<MulOp, mlir::arith::MulIOp, mlir::arith::MulFOp>,
      BinaryArithOpLowering<OrOp, mlir::arith::OrIOp>,
      BinaryArithOpLowering<RShiftOp, mlir::arith::ShRSIOp>,
      BinaryArithOpLowering<SubOp, mlir::arith::SubIOp, mlir::arith::SubFOp>,
      BinaryArithOpLowering<TrueDivOp, void, mlir::arith::DivFOp>,
      BinaryArithOpLowering<XorOp, mlir::arith::XOrIOp>,
      ComparisonOpLowering<EqOp, mlir::arith::CmpIPredicate::eq,
                           mlir::arith::CmpFPredicate::OEQ>,
      ComparisonOpLowering<GeOp, mlir::arith::CmpIPredicate::sge,
                           mlir::arith::CmpFPredicate::OGE>,
      ComparisonOpLowering<GtOp, mlir::arith::CmpIPredicate::sgt,
                           mlir::arith::CmpFPredicate::OGT>,
      ComparisonOpLowering<LeOp, mlir::arith::CmpIPredicate::sle,
                           mlir::arith::CmpFPredicate::OLE>,
      ComparisonOpLowering<LtOp, mlir::arith::CmpIPredicate::slt,
                           mlir::arith::CmpFPredicate::OLT>,
      ComparisonOpLowering<NeOp, mlir::arith::CmpIPredicate::ne,
                           mlir::arith::CmpFPredicate::UNE>,
      ModOpLowering, NegOpLowering, NotOpLowering>(patterns.getContext());
}

} // namespace trident::tvm_ffi::arith
