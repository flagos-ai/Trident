// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Conversion/TVMFFIAPIToABI/TVMFFIAPIToABI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/API/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFITypes.h"
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
#include <mlir/IR/OperationSupport.h>
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

namespace trident::conversion {
namespace abi = tvm_ffi::abi;
using namespace tvm_ffi;
using namespace tvm_ffi::api;
#define GEN_PASS_DEF_CONVERTTVMFFIAPITOABI
#include "trident/core/Conversion/Passes.h.inc"

namespace {

static mlir::Value
createGlobalFunctionCall(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Location location, llvm::StringRef name,
                         mlir::Type resultType, mlir::ValueRange arguments) {
  mlir::Value const handle = abi::FunctionGetGlobalOp::create(
      rewriter, location, abi::FunctionType::get(rewriter.getContext()), name);
  mlir::Value const result = abi::FunctionCallOp::create(
      rewriter, location, resultType, handle, arguments);
  abi::ObjectDecRefOp::create(rewriter, location, handle);
  return result;
}

static void createObjectDecRefs(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location location, mlir::TypeRange types,
                                mlir::ValueRange values) {
  abi::DeferOp cleanup = abi::DeferOp::create(rewriter, location);
  rewriter.setInsertionPointToEnd(cleanup.addEntryBlock(rewriter));
  for (auto [type, value] : llvm::make_filter_range(
           llvm::zip_equal(types, values), [](auto it) -> bool {
             auto [type, _] = it;
             return mlir::isa<TVMFFICOpaqueABIType>(type);
           })) {
    abi::ObjectDecRefOp::create(rewriter, location, value);
  }
  abi::DeferYieldOp::create(rewriter, location);
}

static void createObjectIncRefs(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location location, mlir::TypeRange types,
                                mlir::ValueRange values) {
  for (auto [type, value] : llvm::make_filter_range(
           llvm::zip_equal(types, values), [](auto it) -> bool {
             auto [type, _] = it;
             return mlir::isa<TVMFFICOpaqueABIType>(type);
           })) {
    abi::ObjectIncRefOp::create(rewriter, location, value);
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
    mlir::Value const index = abi::ToOp::create(
        rewriter, op.getLoc(), abi::IntType::get(rewriter.getContext()),
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
        abi::ConstantRawStrOp::create(rewriter, op.getLoc(), op.getValueAttr());
    abi::UpcastOp const newOp = abi::UpcastOp::create(
        rewriter, op.getLoc(), getTypeConverter()->convertType(op.getType()),
        raw);
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
      rewriter.replaceOpWithNewOp<abi::DowncastOp>(
          op, mlir::TypeRange{target}, mlir::ValueRange{adaptor.getValue()},
          op->getAttrs());
    }
    return mlir::success();
  }
};

// Exports retain each original Opaque operand, including repeated occurrences.
// Keep this boundary rule separate from the ownership of producer results.
template <typename Op, typename TargetOp = Op>
class ConvertExportOp final : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    createObjectIncRefs(rewriter, op.getLoc(), op->getOperandTypes(),
                        adaptor.getOperands());
    rewriter.replaceOpWithNewOp<TargetOp>(
        op, mlir::TypeRange{}, adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

class ConvertExecuteRegionOp final
    : public mlir::OpConversionPattern<mlir::scf::ExecuteRegionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ExecuteRegionOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types)) ||
        mlir::failed(rewriter.convertRegionTypes(&op.getRegion(),
                                                 *getTypeConverter()))) {
      return mlir::failure();
    }
    mlir::scf::ExecuteRegionOp replacement = mlir::scf::ExecuteRegionOp::create(
        rewriter, op.getLoc(), types, op.getNoInlineAttr());
    replacement->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getRegion(), replacement.getRegion(),
                                replacement.getRegion().end());
    rewriter.setInsertionPointAfter(replacement);
    createObjectDecRefs(rewriter, op.getLoc(), op.getResultTypes(),
                        replacement.getResults());
    rewriter.replaceOp(op, replacement.getResults());
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
template <typename Op, typename TargetOp = Op, bool OwnsResults = false>
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
      TargetOp newOp = TargetOp::create(rewriter, op.getLoc(), types,
                                        adaptor.getOperands(), op->getAttrs());
      rewriter.setInsertionPointAfter(newOp);
      createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(),
                          newOp->getResults());
      rewriter.replaceOp(op, newOp->getResults());
    } else {
      rewriter.replaceOpWithNewOp<TargetOp>(op, types, adaptor.getOperands(),
                                            op->getAttrs());
    }
    return mlir::success();
  }
};

class ConvertGlobalCallOp final
    : public mlir::OpConversionPattern<GlobalCallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GlobalCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type const resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) {
      return mlir::failure();
    }
    mlir::Value const result =
        createGlobalFunctionCall(rewriter, op.getLoc(), op.getName(),
                                 resultType, adaptor.getArguments());
    result.getDefiningOp()->setAttrs(op->getDiscardableAttrDictionary());
    createObjectDecRefs(rewriter, op.getLoc(), op->getResultTypes(), {result});
    rewriter.replaceOp(op, {result});
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
          mlir::Value const boxed = abi::ToOp::create(
              rewriter, op.getLoc(), abi::IntType::get(rewriter.getContext()),
              offset);
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
    if (!mlir::isa<mlir::scf::ExecuteRegionOp, mlir::scf::ForOp,
                   mlir::scf::IfOp, mlir::scf::WhileOp>(op->getParentOp())) {
      return rewriter.notifyMatchFailure(op, "unsupported region boundary");
    }
    createObjectIncRefs(rewriter, op.getLoc(), op.getOperandTypes(),
                        adaptor.getOperands());
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
        op, mlir::TypeRange{}, adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

template <typename SourceOp, typename TargetOp>
class ConvertFunctionOp final : public mlir::OpConversionPattern<SourceOp> {
public:
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type const type =
        this->getTypeConverter()->convertType(op.getFunctionType());
    if (!type || mlir::failed(rewriter.convertRegionTypes(
                     &op.getBody(), *this->getTypeConverter()))) {
      return mlir::failure();
    }
    mlir::OperationState state(op.getLoc(), TargetOp::getOperationName());
    state.addAttributes(op->getAttrs());
    state.attributes.set(op.getFunctionTypeAttrName(),
                         mlir::TypeAttr::get(type));
    state.addRegion();
    TargetOp replacement = mlir::cast<TargetOp>(rewriter.create(state));
    rewriter.inlineRegionBefore(op.getBody(), replacement.getBody(),
                                replacement.getBody().end());
    rewriter.replaceOp(op, replacement->getResults());
    return mlir::success();
  }
};

class ConvertTVMFFIAPIToABIPass final
    : public impl::ConvertTVMFFIAPIToABIBase<ConvertTVMFFIAPIToABIPass> {
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    mlir::MLIRContext *context = op->getContext();
    mlir::TypeConverter converter;
    // TODO: Preserve finite Union/Str alternatives across type erasure for
    // packed ABI argument validation and success-only return checks.
    mlir::RewritePatternSet conversions(context);
    mlir::ConversionTarget target(*context);
    populateConvertTVMFFIAPIToABIConversionPatterns(target, converter,
                                                    conversions);

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

void populateConvertTVMFFIAPIToABIConversionPatterns(
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
  converter.addConversion(
      [&converter, context](tvm_ffi::monad::ResultType type) -> mlir::Type {
        return tvm_ffi::monad::ResultType::get(
            context, mlir::cast<TVMFFICABIType>(
                         converter.convertType(type.getPayloadType())));
      });
  converter.addConversion(
      [context](AnyType) -> mlir::Type { return abi::AnyType::get(context); });
  converter.addConversion([context](ArrayType) -> mlir::Type {
    return abi::ArrayType::get(context);
  });
  converter.addConversion([context](ExceptionType) -> mlir::Type {
    return abi::ExceptionType::get(context);
  });
  converter.addConversion([context](FunctionType) -> mlir::Type {
    return abi::FunctionType::get(context);
  });
  converter.addConversion(
      [context](StrType) -> mlir::Type { return abi::AnyType::get(context); });
  converter.addConversion([context](TensorType) -> mlir::Type {
    return abi::TensorType::get(context);
  });
  converter.addConversion([context](TupleType) -> mlir::Type {
    return abi::ArrayType::get(context);
  });
  converter.addConversion([context](UnionType) -> mlir::Type {
    return abi::AnyType::get(context);
  });

  patterns.add<
      ConvertArrayConstructionOp<ArrayCreateOp>,
      ConvertArrayConstructionOp<TupleConstructOp>, ConvertArrayGetOp,
      ConvertArraySizeOp, ConvertConstantStrOp, ConvertDowncastOp,
      ConvertExportOp<ReturnOp, abi::ReturnOp>,
      ConvertExportOp<mlir::func::ReturnOp>,
      ConvertGenericOp<CallOp, abi::CallOp, true>,
      ConvertGenericOp<EqOp, abi::EqOp>,
      ConvertGenericOp<ExceptionOp, abi::ExceptionOp, true>,
      ConvertGenericOp<FunctionCallOp, abi::FunctionCallOp, true>,
      ConvertGenericOp<DirectAllocationOp, abi::DirectAllocationOp, true>,
      ConvertGenericOp<FunctionGetGlobalOp, abi::FunctionGetGlobalOp, true>,
      ConvertGenericOp<GetOp, abi::GetOp>, ConvertGenericOp<IsaOp, abi::IsaOp>,
      ConvertGenericOp<tvm_ffi::monad::FailOp>,
      ConvertGenericOp<tvm_ffi::monad::PureOp>,
      ConvertGenericOp<tvm_ffi::monad::UnpackOp>,

      ConvertGenericOp<PackCallOp, abi::PackCallOp, true>,
      ConvertGenericOp<PackSizeOp, abi::PackSizeOp>,
      ConvertGenericOp<PackGetOp, abi::PackGetOp>,
      ConvertGenericOp<TensorCloneOp, abi::TensorCloneOp, true>,
      ConvertGenericOp<TensorCopyOp, abi::TensorCopyOp>,
      ConvertGenericOp<TensorDTypeOp, abi::TensorDTypeOp>,
      ConvertGenericOp<TensorDeviceOp, abi::TensorDeviceOp>,
      ConvertGenericOp<TensorDimOp, abi::TensorDimOp>,
      ConvertGenericOp<TensorLiteralOp, abi::TensorLiteralOp, true>,
      ConvertGenericOp<TensorSizeOp, abi::TensorSizeOp>,
      ConvertGenericOp<TensorStorageOffsetOp, abi::TensorStorageOffsetOp>,
      ConvertGenericOp<TensorStrideOp, abi::TensorStrideOp>,
      ConvertGenericOp<UpcastOp, abi::UpcastOp>,
      ConvertGenericOp<mlir::func::CallOp, mlir::func::CallOp, true>,
      ConvertGlobalCallOp, ConvertTupleUnconstructOp>(converter, context);
  patterns.add<ConvertConditionOp, ConvertExecuteRegionOp, ConvertForOp,
               ConvertIfOp, ConvertWhileOp, ConvertYieldOp>(converter, context);
  patterns.add<ConvertFunctionOp<FuncOp, abi::FuncOp>,
               ConvertFunctionOp<PackFuncOp, abi::PackFuncOp>>(converter,
                                                               context);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, converter);
  mlir::cf::populateCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                           target);
  target.addIllegalDialect<TVMFFIAPIDialect>();
  target.addDynamicallyLegalOp<tvm_ffi::monad::FailOp, tvm_ffi::monad::PureOp,
                               tvm_ffi::monad::UnpackOp>(
      [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal([&](mlir::Operation *operation) -> bool {
    return converter.isLegal(operation) &&
           llvm::all_of(operation->getRegions(),
                        [&](mlir::Region &region) -> bool {
                          return converter.isLegal(&region);
                        });
  });
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::Operation *operation) -> bool {
        mlir::FunctionOpInterface function =
            mlir::cast<mlir::FunctionOpInterface>(operation);
        mlir::Region &body = function.getFunctionBody();
        return converter.isLegal(function.getFunctionType()) &&
               (body.empty() ||
                converter.isLegal(body.front().getArgumentTypes()));
      });
}

} // namespace trident::conversion
