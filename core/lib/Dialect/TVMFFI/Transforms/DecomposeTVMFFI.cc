// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/DecomposeTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cassert>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <utility>

namespace trident::tvm_ffi {
#define GEN_PASS_DEF_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

namespace {

static mlir::Value
createGlobalFunctionCall(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Location location, llvm::StringRef name,
                         mlir::Type resultType, mlir::ValueRange arguments) {
  mlir::Value const handle = FunctionGetGlobalOp::create(
      rewriter, location, CFunctionType::get(rewriter.getContext()), name);
  mlir::Value const result =
      FunctionCallOp::create(rewriter, location, resultType, handle, arguments);
  ObjectDecRefOp::create(rewriter, location, handle);
  return result;
}

static void createObjectDecRefs(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location location, mlir::TypeRange types,
                                mlir::ValueRange values) {
  DeferOp cleanup = DeferOp::create(rewriter, location);
  rewriter.setInsertionPointToEnd(cleanup.addEntryBlock(rewriter));
  for (auto [type, value] : llvm::make_filter_range(
           llvm::zip_equal(types, values), [](auto it) -> bool {
             auto [type, _] = it;
             return mlir::isa<TVMFFICOpaqueABIType>(type);
           })) {
    ObjectDecRefOp::create(rewriter, location, value);
  }
  DeferYieldOp::create(rewriter, location);
}

static void createObjectIncRefs(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location location, mlir::TypeRange types,
                                mlir::ValueRange values) {
  for (auto [type, value] : llvm::make_filter_range(
           llvm::zip_equal(types, values), [](auto it) -> bool {
             auto [type, _] = it;
             return mlir::isa<TVMFFICOpaqueABIType>(type);
           })) {
    ObjectIncRefOp::create(rewriter, location, value);
  }
}

// Arrays and tuples share the runtime representation and construction protocol.
template <typename Op>
class ConvertArrayConstructionOp final : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const result = createGlobalFunctionCall(
        rewriter, op.getLoc(), "ffi.Array",
        this->getTypeConverter()->convertType(op.getType()),
        adaptor.getOperands());
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(), {result});
    rewriter.replaceOp(op, {result});
    return mlir::success();
  }
};

class ConvertArrayGetOp final : public mlir::OpConversionPattern<ArrayGetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const index =
        ToOp::create(rewriter, op.getLoc(), IntType::get(rewriter.getContext()),
                     adaptor.getIndex());
    mlir::Value const result = createGlobalFunctionCall(
        rewriter, op.getLoc(), "ffi.ArrayGetItem",
        getTypeConverter()->convertType(op.getType()),
        mlir::ValueRange{adaptor.getContainer(), index});
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(), {result});
    rewriter.replaceOp(op, {result});
    return mlir::success();
  }
};

class ConvertArraySizeOp final : public mlir::OpConversionPattern<ArraySizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArraySizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const result = createGlobalFunctionCall(
        rewriter, op.getLoc(), "ffi.ArraySize",
        getTypeConverter()->convertType(op.getType()), adaptor.getContainer());
    rewriter.replaceOp(op, {result});
    return mlir::success();
  }
};

class ConvertConditionOp final
    : public mlir::OpConversionPattern<mlir::scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ConditionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Both outcomes forward the payload; the native predicate is borrowed.
    createObjectIncRefs(rewriter, op.getLoc(), op.getArgs().getTypes(),
                        adaptor.getArgs());
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        op, mlir::TypeRange{}, adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

class ConvertConstantStrOp final
    : public mlir::OpConversionPattern<ConstantStrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConstantStrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const raw =
        ConstantRawStrOp::create(rewriter, op.getLoc(), op.getValueAttr());
    UpcastOp const newOp =
        UpcastOp::create(rewriter, op.getLoc(),
                         getTypeConverter()->convertType(op.getType()), raw);
    rewriter.setInsertionPointAfter(newOp);
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(),
                        newOp->getResults());
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

class ConvertDowncastOp final : public mlir::OpConversionPattern<DowncastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DowncastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type const target = getTypeConverter()->convertType(op.getType());
    // TODO: Preserve finite Union/Str type constraints before erasing them to
    // CAny and materialize runtime type checks.
    if (adaptor.getValue().getType() == target) {
      rewriter.replaceOp(op, adaptor.getValue());
    } else {
      rewriter.replaceOpWithNewOp<DowncastOp>(
          op, mlir::TypeRange{target}, mlir::ValueRange{adaptor.getValue()},
          op->getAttrs());
    }
    return mlir::success();
  }
};

// Exports retain each original Opaque operand, including repeated occurrences.
// Keep this boundary rule separate from the ownership of producer results.
template <typename Op>
class ConvertExportOp final : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    createObjectIncRefs(rewriter, op.getLoc(), op->getOperandTypes(),
                        adaptor.getOperands());
    rewriter.replaceOpWithNewOp<Op>(op, mlir::TypeRange{},
                                    adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

class ConvertForOp final : public mlir::OpConversionPattern<mlir::scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> const argumentTypes(
        op.getInitArgs().getTypes());
    createObjectIncRefs(rewriter, op.getLoc(), op.getInitArgs().getTypes(),
                        adaptor.getInitArgs());
    if (mlir::failed(rewriter.convertRegionTypes(&op.getRegion(),
                                                 *getTypeConverter()))) {
      return mlir::failure();
    }
    mlir::scf::ForOp newOp = mlir::scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs(), nullptr, op.getUnsignedCmp());
    newOp->setAttrs(op->getAttrs());
    rewriter.eraseBlock(newOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.setInsertionPointToStart(newOp.getBody());
    createObjectDecRefs(rewriter, op.getLoc(), argumentTypes,
                        newOp.getRegionIterArgs());
    rewriter.setInsertionPointAfter(newOp);
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(),
                        newOp->getResults());
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

// Only explicitly registered regionless operations use this representation-only
// rewrite. Producer registrations explicitly enable result ownership.
template <typename Op, bool OwnsResults = false>
class ConvertGenericOp final : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(this->getTypeConverter()->convertTypes(
            op->getResultTypes(), types))) {
      return mlir::failure();
    }
    if constexpr (OwnsResults) {
      Op newOp = Op::create(rewriter, op.getLoc(), types, adaptor.getOperands(),
                            op->getAttrs());
      rewriter.setInsertionPointAfter(newOp);
      createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(),
                          newOp->getResults());
      rewriter.replaceOp(op, newOp->getResults());
    } else {
      rewriter.replaceOpWithNewOp<Op>(op, types, adaptor.getOperands(),
                                      op->getAttrs());
    }
    return mlir::success();
  }
};

class ConvertIfOp final : public mlir::OpConversionPattern<mlir::scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types))) {
      return mlir::failure();
    }
    mlir::scf::IfOp newOp = mlir::scf::IfOp::create(
        rewriter, op.getLoc(), types, adaptor.getCondition(), false, false);
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());
    rewriter.setInsertionPointAfter(newOp);
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(),
                        newOp->getResults());
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

class ConvertTupleUnconstructOp final
    : public mlir::OpConversionPattern<TupleUnconstructOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TupleUnconstructOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> const results = llvm::map_to_vector(
        llvm::enumerate(op.getResultTypes()), [&](auto entry) -> mlir::Value {
          auto [index, type] = entry;
          mlir::Value const offset = mlir::arith::ConstantIntOp::create(
              rewriter, op.getLoc(), static_cast<int64_t>(index), 64);
          mlir::Value const boxed =
              ToOp::create(rewriter, op.getLoc(),
                           IntType::get(rewriter.getContext()), offset);
          return createGlobalFunctionCall(
              rewriter, op.getLoc(), "ffi.ArrayGetItem",
              getTypeConverter()->convertType(type),
              mlir::ValueRange{adaptor.getContainer(), boxed});
        });
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(), results);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

class ConvertWhileOp final
    : public mlir::OpConversionPattern<mlir::scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> const beforeTypes(op.getInits().getTypes());
    llvm::SmallVector<mlir::Type> const afterTypes(op.getResultTypes());
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(getTypeConverter()->convertTypes(afterTypes, types))) {
      return mlir::failure();
    }
    createObjectIncRefs(rewriter, op.getLoc(), op.getInits().getTypes(),
                        adaptor.getInits());
    if (mlir::failed(rewriter.convertRegionTypes(&op.getBefore(),
                                                 *getTypeConverter())) ||
        mlir::failed(
            rewriter.convertRegionTypes(&op.getAfter(), *getTypeConverter()))) {
      return mlir::failure();
    }
    mlir::scf::WhileOp newOp = mlir::scf::WhileOp::create(
        rewriter, op.getLoc(), types, adaptor.getInits());
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getBefore(), newOp.getBefore(),
                                newOp.getBefore().end());
    rewriter.inlineRegionBefore(op.getAfter(), newOp.getAfter(),
                                newOp.getAfter().end());
    rewriter.setInsertionPointToStart(newOp.getBeforeBody());
    createObjectDecRefs(rewriter, op.getLoc(), beforeTypes,
                        newOp.getBeforeArguments());
    rewriter.setInsertionPointToStart(newOp.getAfterBody());
    createObjectDecRefs(rewriter, op.getLoc(), afterTypes,
                        newOp.getAfterArguments());
    rewriter.setInsertionPointAfter(newOp);
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(),
                        newOp->getResults());
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

class ConvertYieldOp final
    : public mlir::OpConversionPattern<mlir::scf::YieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::WhileOp>(
            op->getParentOp())) {
      return rewriter.notifyMatchFailure(op, "unsupported region boundary");
    }
    createObjectIncRefs(rewriter, op.getLoc(), op.getOperandTypes(),
                        adaptor.getOperands());
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
        op, mlir::TypeRange{}, adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

class DecomposeTVMFFIPass final
    : public impl::DecomposeTVMFFIBase<DecomposeTVMFFIPass> {
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    mlir::MLIRContext *context = op->getContext();
    mlir::TypeConverter converter;
    // TODO: Preserve finite Union/Str alternatives across type erasure for
    // packed ABI argument validation and success-only return checks.
    mlir::RewritePatternSet conversions(context);
    mlir::ConversionTarget target(*context);
    populateDecomposeTVMFFIConversionPatterns(target, converter, conversions);

    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(conversions)))) {
      signalPassFailure();
      return;
    }
    // Attributes can contain nested types unrelated to operands or results.
    // Rewrite these only after operation conversion, when ownership no longer
    // needs them.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](mlir::Type type) -> mlir::Type {
      return converter.convertType(type);
    });
    replacer.recursivelyReplaceElementsIn(op);
  }
};

} // namespace

void populateDecomposeTVMFFIConversionPatterns(
    mlir::ConversionTarget &target, mlir::TypeConverter &converter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  // Register the identity fallback first: specific conversions take priority.
  converter.addConversion([](mlir::Type type) -> mlir::Type { return type; });
  converter.addConversion(
      [&converter](mlir::FunctionType type) -> std::optional<mlir::Type> {
        llvm::SmallVector<mlir::Type> inputs;
        llvm::SmallVector<mlir::Type> outputs;
        if (mlir::failed(converter.convertTypes(type.getInputs(), inputs)) ||
            mlir::failed(converter.convertTypes(type.getResults(), outputs))) {
          return std::nullopt;
        }
        return mlir::FunctionType::get(type.getContext(), inputs, outputs);
      });
  converter.addConversion([&converter, context](ResultType type) -> mlir::Type {
    return ResultType::get(context,
                           mlir::cast<TVMFFICABIType>(
                               converter.convertType(type.getPayloadType())));
  });
  converter.addConversion(
      [context](AnyType) -> mlir::Type { return CAnyType::get(context); });
  converter.addConversion(
      [context](ArrayType) -> mlir::Type { return CArrayType::get(context); });
  converter.addConversion([context](ExceptionType) -> mlir::Type {
    return CExceptionType::get(context);
  });
  converter.addConversion([context](FunctionType) -> mlir::Type {
    return CFunctionType::get(context);
  });
  converter.addConversion(
      [context](StrType) -> mlir::Type { return CAnyType::get(context); });
  converter.addConversion([context](TensorType) -> mlir::Type {
    return CTensorType::get(context);
  });
  converter.addConversion(
      [context](TupleType) -> mlir::Type { return CArrayType::get(context); });
  converter.addConversion(
      [context](UnionType) -> mlir::Type { return CAnyType::get(context); });

  patterns.add<
      ConvertArrayConstructionOp<ArrayCreateOp>,
      ConvertArrayConstructionOp<TupleConstructOp>, ConvertArrayGetOp,
      ConvertArraySizeOp, ConvertConstantStrOp, ConvertDowncastOp,
      ConvertExportOp<ReturnOp>, ConvertExportOp<mlir::func::ReturnOp>,
      ConvertGenericOp<CallOp, true>, ConvertGenericOp<EqOp>,
      ConvertGenericOp<ExceptionOp, true>,
      ConvertGenericOp<FunctionCallOp, true>,
      ConvertGenericOp<FunctionGetGlobalOp, true>, ConvertGenericOp<GetOp>,
      ConvertGenericOp<IsaOp>, ConvertGenericOp<MonadFailOp>,
      ConvertGenericOp<MonadPureOp>, ConvertGenericOp<MonadUnpackOp>,
      ConvertGenericOp<ObjectDecRefOp>, ConvertGenericOp<ObjectIncRefOp>,
      ConvertGenericOp<PackCallOp, true>, ConvertGenericOp<PackGetOp>,
      ConvertGenericOp<TensorCloneOp, true>, ConvertGenericOp<TensorCopyOp>,
      ConvertGenericOp<TensorDTypeOp>, ConvertGenericOp<TensorDeviceOp>,
      ConvertGenericOp<TensorDimOp>, ConvertGenericOp<TensorLiteralOp, true>,
      ConvertGenericOp<TensorSizeOp>, ConvertGenericOp<TensorStorageOffsetOp>,
      ConvertGenericOp<TensorStrideOp>, ConvertGenericOp<UpcastOp>,
      ConvertGenericOp<mlir::func::CallOp, true>, ConvertTupleUnconstructOp>(
      converter, context);
  patterns.add<ConvertConditionOp, ConvertForOp, ConvertIfOp, ConvertWhileOp,
               ConvertYieldOp>(converter, context);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                                 converter);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<PackFuncOp>(patterns,
                                                                     converter);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, converter);
  mlir::cf::populateCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                           target);
  target.addIllegalOp<ArrayCreateOp, ArrayGetOp, ArraySizeOp, ConstantStrOp,
                      TupleConstructOp, TupleUnconstructOp>();
  target.markUnknownOpDynamicallyLegal([&](mlir::Operation *operation) -> bool {
    return converter.isLegal(operation) &&
           llvm::all_of(operation->getRegions(),
                        [&](mlir::Region &region) -> bool {
                          return converter.isLegal(&region);
                        });
  });
  target.addDynamicallyLegalOp<FuncOp, PackFuncOp, mlir::func::FuncOp>(
      [&](mlir::Operation *operation) -> bool {
        mlir::FunctionOpInterface function =
            mlir::cast<mlir::FunctionOpInterface>(operation);
        mlir::Region &body = function.getFunctionBody();
        return converter.isLegal(function.getFunctionType()) &&
               (body.empty() ||
                converter.isLegal(body.front().getArgumentTypes()));
      });
}

} // namespace trident::tvm_ffi
