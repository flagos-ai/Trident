// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/LowerTVMFFIMonad.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <type_traits>
#include <utility>

namespace trident::tvm_ffi {
#define GEN_PASS_DEF_LOWERTVMFFIMONAD
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

namespace {

TupleType resultTupleType(mlir::MLIRContext *context) {
  return TupleType::get(context,
                        {BoolType::get(context), AnyType::get(context)});
}

// Tuple construction borrows its elements; low-level pure instead transfers
// payload references. Release those inputs after the container has retained
// them.
mlir::Value makeTuple(mlir::OpBuilder &builder, mlir::Location loc,
                      bool success, mlir::Value payload, bool cleanup) {
  mlir::Value const flag = ConstantBoolOp::create(builder, loc, success);
  payload = builder.createOrFold<UpcastOp>(
      loc, AnyType::get(builder.getContext()), payload);
  TupleConstructOp tuple = TupleConstructOp::create(
      builder, loc, resultTupleType(builder.getContext()),
      mlir::ValueRange{flag, payload});
  if (cleanup) {
    tuple->setAttr(kCleanupAttributeName, builder.getUnitAttr());
  }
  return tuple;
}

class LowerPureOp final : public mlir::OpConversionPattern<MonadPureOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(MonadPureOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> results;
    for (mlir::Value payload : adaptor.getValues()) {
      payload = rewriter.createOrFold<UpcastOp>(
          op.getLoc(), AnyType::get(rewriter.getContext()), payload);
      results.push_back(makeTuple(rewriter, op.getLoc(), true, payload,
                                  op->hasAttr(kCleanupAttributeName)));
      if (!op->hasAttr(kRetainAttributeName)) {
        ObjectDecRefOp::create(rewriter, op.getLoc(), payload);
      }
    }
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

class LowerFailOp final : public mlir::OpConversionPattern<MonadFailOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(MonadFailOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const none = ConstantNoneOp::create(rewriter, op.getLoc());
    llvm::SmallVector<mlir::Value> const results = llvm::map_to_vector(
        op.getOutputs(), [&rewriter, &op, &none](mlir::Value) -> mlir::Value {
          return makeTuple(rewriter, op.getLoc(), false, none, true);
        });
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

class LowerUnpackOp final : public mlir::OpConversionPattern<MonadUnpackOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(MonadUnpackOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TupleUnconstructOp tuple = TupleUnconstructOp::create(
        rewriter, op.getLoc(),
        mlir::TypeRange{BoolType::get(rewriter.getContext()),
                        AnyType::get(rewriter.getContext())},
        adaptor.getInput());
    tuple->setAttr(kCleanupAttributeName, rewriter.getUnitAttr());
    mlir::Value const success = GetOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(), tuple.getElements()[0]);
    rewriter.replaceOp(op, {success, tuple.getElements()[1]});
    return mlir::success();
  }
};

class LowerCallOp final : public mlir::OpConversionPattern<MonadCallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MonadCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    CallOp call =
        CallOp::create(rewriter, loc, BoolType::get(rewriter.getContext()),
                       AnyType::get(rewriter.getContext()), op.getCalleeAttr(),
                       adaptor.getOperands(), op.getArgAttrsAttr(), nullptr);
    call->setAttrs(op->getAttrs());
    call->removeAttr(kCleanupAttributeName);
    if (mlir::ArrayAttr const attributes = op.getResAttrsAttr();
        attributes && !attributes.empty()) {
      call.setResAttrsAttr(rewriter.getArrayAttr(
          {rewriter.getDictionaryAttr({}), attributes[0]}));
    }
    TupleConstructOp const tuple =
        rewriter.replaceOpWithNewOp<TupleConstructOp>(
            op, resultTupleType(rewriter.getContext()), call.getResults());
    if (op->hasAttr(kCleanupAttributeName)) {
      tuple->setAttr(kCleanupAttributeName, rewriter.getUnitAttr());
    }
    rewriter.setInsertionPointAfter(tuple);
    ObjectDecRefOp::create(rewriter, loc, call.getValue());
    return mlir::success();
  }
};

class LowerBindOp final : public mlir::OpConversionPattern<MonadBindOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MonadBindOp bind, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter const &converter = *getTypeConverter();
    llvm::SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(
            converter.convertTypes(bind.getResultTypes(), resultTypes)) ||
        mlir::failed(rewriter.convertRegionTypes(&bind.getBody(), converter))) {
      return mlir::failure();
    }
    mlir::Location const loc = bind.getLoc();
    mlir::Value ready = mlir::arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    llvm::SmallVector<mlir::Value> const unpacked = llvm::map_to_vector(
        adaptor.getInputs(), [&](mlir::Value input) -> mlir::Value {
          TupleUnconstructOp unpack = TupleUnconstructOp::create(
              rewriter, loc,
              mlir::TypeRange{BoolType::get(rewriter.getContext()),
                              AnyType::get(rewriter.getContext())},
              input);
          unpack->setAttr(kCleanupAttributeName, rewriter.getUnitAttr());
          mlir::Value const success = GetOp::create(
              rewriter, loc, rewriter.getI1Type(), unpack.getElements()[0]);
          ready =
              rewriter.createOrFold<mlir::arith::AndIOp>(loc, ready, success);
          return unpack.getElements()[1];
        });
    mlir::scf::IfOp branch = mlir::scf::IfOp::create(
        rewriter, loc, resultTypes, ready, !resultTypes.empty());
    if (resultTypes.empty()) {
      rewriter.eraseOp(branch.thenBlock()->getTerminator());
    }
    if (mlir::Attribute const cleanup = bind->getAttr(kCleanupAttributeName)) {
      branch->setAttr(kCleanupAttributeName, cleanup);
    }
    rewriter.setInsertionPointToStart(branch.thenBlock());
    mlir::Block &body = *bind.getEntryBlock();
    llvm::SmallVector<mlir::Value> const payloads =
        llvm::map_to_vector(llvm::zip_equal(unpacked, body.getArguments()),
                            [&](const auto &entry) -> mlir::Value {
                              auto [value, argument] = entry;
                              return rewriter.createOrFold<DowncastOp>(
                                  loc, argument.getType(), value);
                            });
    // Move rather than clone the continuation, as in SCF structural conversion.
    // Its yield is converted separately using the remapped operands.
    rewriter.mergeBlocks(&body, branch.thenBlock(), payloads);
    if (!resultTypes.empty()) {
      rewriter.setInsertionPointToStart(branch.elseBlock());
      MonadFailOp failed =
          MonadFailOp::create(rewriter, loc, bind.getResultTypes());
      failed->setAttr(kCleanupAttributeName, rewriter.getUnitAttr());
      mlir::scf::YieldOp const exit =
          mlir::scf::YieldOp::create(rewriter, loc, failed.getResults());
      exit->setAttr(kRetainAttributeName, rewriter.getUnitAttr());
    }
    rewriter.replaceOp(bind, branch.getResults());
    return mlir::success();
  }
};

// Failure previously held no references. Its new container is local-owned;
// establish an export reference even for an otherwise unmarked transfer exit.
void retainFailureExports(mlir::Operation *op, mlir::ValueRange operands,
                          mlir::ConversionPatternRewriter &rewriter) {
  if (op->hasAttr(kRetainAttributeName)) {
    return;
  }
  for (auto [original, converted] :
       llvm::zip_equal(op->getOperands(), operands)) {
    if (original.getDefiningOp<MonadFailOp>()) {
      mlir::Value const any =
          UpcastOp::create(rewriter, op->getLoc(),
                           AnyType::get(rewriter.getContext()), converted);
      ObjectIncRefOp::create(rewriter, op->getLoc(), any);
    }
  }
}

class LowerYieldOp final : public mlir::OpConversionPattern<MonadYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MonadYieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    retainFailureExports(op, adaptor.getOperands(), rewriter);
    mlir::DictionaryAttr const attributes = op->getAttrDictionary();
    mlir::scf::YieldOp const replacement =
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op,
                                                        adaptor.getValues());
    replacement->setAttrs(attributes);
    return mlir::success();
  }
};

// These regionless operations keep their semantics; only operands and results
// change. Region-bearing operations use their own structural conversions below.
template <typename OpTy>
class GenericOpPattern final : public mlir::OpConversionPattern<OpTy> {
public:
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<OpTy, mlir::func::ReturnOp> ||
                  std::is_same_v<OpTy, mlir::scf::YieldOp>) {
      retainFailureExports(op, adaptor.getOperands(), rewriter);
    }
    mlir::FailureOr<mlir::Operation *> replacement = mlir::convertOpResultTypes(
        op, adaptor.getOperands(), *this->getTypeConverter(), rewriter);
    if (mlir::failed(replacement)) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, (*replacement)->getResults());
    return mlir::success();
  }
};

class MonadTypeConverter final : public mlir::TypeConverter {
public:
  explicit MonadTypeConverter(mlir::MLIRContext *context) {
    addConversion([](mlir::Type type) -> mlir::Type { return type; });
    addConversion([context](ResultType) -> mlir::Type {
      return resultTupleType(context);
    });
    addConversion([this, context](mlir::FunctionType type) -> mlir::Type {
      llvm::SmallVector<mlir::Type> inputs, outputs;
      if (mlir::failed(convertTypes(type.getInputs(), inputs)) ||
          mlir::failed(convertTypes(type.getResults(), outputs))) {
        return {};
      }
      return mlir::FunctionType::get(context, inputs, outputs);
    });
  }
};

class LowerTVMFFIMonadPass final
    : public impl::LowerTVMFFIMonadBase<LowerTVMFFIMonadPass> {
  void runOnOperation() final {
    mlir::Operation *op = getOperation();
    MonadTypeConverter converter(&getContext());
    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<MonadCallOp, MonadBindOp, MonadFailOp, MonadPureOp,
                        MonadUnpackOp, MonadYieldOp>();
    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *operation) -> bool {
          return converter.isLegal(operation) &&
                 llvm::all_of(operation->getRegions(),
                              [&](mlir::Region &region) -> bool {
                                return converter.isLegal(&region);
                              });
        });
    target.addDynamicallyLegalOp<FuncOp, mlir::func::FuncOp>(
        [&](mlir::Operation *operation) -> bool {
          mlir::FunctionOpInterface function =
              mlir::cast<mlir::FunctionOpInterface>(operation);
          mlir::Region &body = function.getFunctionBody();
          return converter.isLegal(function.getFunctionType()) &&
                 (body.empty() ||
                  converter.isLegal(body.front().getArgumentTypes()));
        });
    mlir::RewritePatternSet patterns(&getContext());
    patterns
        .add<LowerBindOp, LowerCallOp, LowerFailOp, LowerPureOp, LowerUnpackOp,
             LowerYieldOp, GenericOpPattern<mlir::func::ReturnOp>>(
            converter, &getContext());
    patterns.add<GenericOpPattern<mlir::func::CallOp>,
                 GenericOpPattern<mlir::scf::YieldOp>>(converter, &getContext(),
                                                       mlir::PatternBenefit(2));
    mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                                   converter);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, patterns, target);
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(converter,
                                                             patterns, target);
    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](mlir::Type type) -> mlir::Type {
      return converter.convertType(type);
    });
    replacer.recursivelyReplaceElementsIn(op);
  }
};

} // namespace
} // namespace trident::tvm_ffi
