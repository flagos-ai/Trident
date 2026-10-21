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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace trident::tvm_ffi {
#define GEN_PASS_DEF_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

namespace {

mlir::LogicalResult validateOwnershipScopes(mlir::Operation *root) {
  mlir::WalkResult const result = root->walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation *op) -> mlir::WalkResult {
        auto isOpaque = [](mlir::Type type) -> bool {
          return mlir::isa<TVMFFICOpaqueABIType>(type);
        };
        if (!llvm::any_of(op->getOperandTypes(), isOpaque) &&
            !llvm::any_of(op->getResultTypes(), isOpaque) &&
            !llvm::any_of(op->getRegions(), [&](mlir::Region &region) -> bool {
              return llvm::any_of(
                  region.getBlocks(), [&](mlir::Block &block) -> bool {
                    return llvm::any_of(block.getArgumentTypes(), isOpaque);
                  });
            })) {
          return mlir::WalkResult::advance();
        }
        // Borrowed operands and exported values also require supported scopes:
        // their lifetimes cannot be inferred across CFG edges or loop
        // iterations.
        auto validateRegion = [&](mlir::Region &region) -> mlir::LogicalResult {
          mlir::Operation *parent = region.getParentOp();
          if (!mlir::isa<mlir::FunctionOpInterface, mlir::scf::IfOp>(parent)) {
            return op->emitError(
                "unsupported region operation for automatic TVM FFI ownership");
          }
          if (!region.empty() && !region.hasOneBlock()) {
            return op->emitError(
                "automatic TVM FFI ownership requires a single-block region");
          }
          return mlir::success();
        };
        for (mlir::Region &region : op->getRegions()) {
          if (mlir::failed(validateRegion(region))) {
            return mlir::WalkResult::interrupt();
          }
        }
        for (mlir::Region *region = op->getParentRegion(); region != nullptr;
             region = region->getParentRegion()) {
          if (mlir::isa<mlir::ModuleOp>(region->getParentOp())) {
            break;
          }
          if (mlir::failed(validateRegion(*region))) {
            return mlir::WalkResult::interrupt();
          }
        }
        return mlir::WalkResult::advance();
      });
  return mlir::failure(result.wasInterrupted());
}

class DecomposeTVMFFITypeConverter final : public mlir::TypeConverter {
public:
  explicit DecomposeTVMFFITypeConverter(mlir::MLIRContext *context) {
    // Register the identity fallback first: specific conversions take priority.
    addConversion([](mlir::Type type) -> mlir::Type { return type; });
    addConversion([this](mlir::FunctionType type) -> mlir::Type {
      llvm::SmallVector<mlir::Type> inputs;
      llvm::SmallVector<mlir::Type> outputs;
      if (mlir::failed(convertTypes(type.getInputs(), inputs)) ||
          mlir::failed(convertTypes(type.getResults(), outputs))) {
        return {};
      }
      return mlir::FunctionType::get(type.getContext(), inputs, outputs);
    });
    addConversion([this, context](ResultType type) -> mlir::Type {
      return ResultType::get(context, mlir::cast<TVMFFICABIType>(
                                          convertType(type.getPayloadType())));
    });
    addConversion(
        [context](AnyType) -> mlir::Type { return CAnyType::get(context); });
    addConversion([context](ArrayType) -> mlir::Type {
      return CArrayType::get(context);
    });
    addConversion([context](ExceptionType) -> mlir::Type {
      return CExceptionType::get(context);
    });
    addConversion([context](FunctionType) -> mlir::Type {
      return CFunctionType::get(context);
    });
    addConversion(
        [context](StrType) -> mlir::Type { return CAnyType::get(context); });
    addConversion([context](TensorType) -> mlir::Type {
      return CTensorType::get(context);
    });
    addConversion([context](TupleType) -> mlir::Type {
      return CArrayType::get(context);
    });
    addConversion(
        [context](UnionType) -> mlir::Type { return CAnyType::get(context); });
  }
};

// Replacement helpers for patterns whose results do not acquire references.
template <typename SourceOp>
class ReplacementConversionPattern
    : public mlir::OpConversionPattern<SourceOp> {
public:
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

protected:
  static void replaceOp(mlir::ConversionPatternRewriter &rewriter, SourceOp op,
                        mlir::ValueRange results) {
    rewriter.replaceOp(op, results);
  }

  static void replaceOp(mlir::ConversionPatternRewriter &rewriter, SourceOp op,
                        mlir::Operation *replacement) {
    mlir::OpBuilder::InsertionGuard const guard(rewriter);
    rewriter.setInsertionPointAfter(replacement);
    replaceOp(rewriter, op, replacement->getResults());
  }

  template <typename Op, typename... Args>
  static Op replaceOpWithNewOp(mlir::ConversionPatternRewriter &rewriter,
                               SourceOp op, Args &&...args) {
    Op replacement =
        Op::create(rewriter, op->getLoc(), std::forward<Args>(args)...);
    replaceOp(rewriter, op, replacement.getOperation());
    return replacement;
  }
};

// Producers explicitly select this base to register ownership in the same
// conversion transaction as replacement.
template <typename SourceOp>
class OwnershipConversionPattern
    : public ReplacementConversionPattern<SourceOp> {
public:
  using ReplacementConversionPattern<SourceOp>::ReplacementConversionPattern;

protected:
  static void replaceOp(mlir::ConversionPatternRewriter &rewriter, SourceOp op,
                        mlir::ValueRange results) {
    assert(op->getNumResults() == results.size() &&
           "ownership replacement requires one value per original result");
    // Ownership belongs to the original Opaque boundary, not to the concrete
    // Base representation or the provenance of an operand.
    llvm::SmallVector<mlir::Value> owned;
    for (mlir::OpResult const result : op->getResults()) {
      uint32_t const index = result.getResultNumber();
      if (mlir::isa<TVMFFICOpaqueABIType>(result.getType())) {
        owned.push_back(results[index]);
      }
    }
    if (!owned.empty()) {
      mlir::OpBuilder::InsertionGuard const guard(rewriter);
      DeferOp cleanup = DeferOp::create(rewriter, op->getLoc());
      mlir::Block *body = rewriter.createBlock(&cleanup.getBody());
      rewriter.setInsertionPointToEnd(body);
      for (mlir::Value const value : llvm::reverse(owned)) {
        ObjectDecRefOp::create(rewriter, cleanup.getLoc(), value);
      }
      DeferYieldOp::create(rewriter, cleanup.getLoc());
    }
    ReplacementConversionPattern<SourceOp>::replaceOp(rewriter, op, results);
  }

  // These overloads deliberately call this class's replaceOp: inherited static
  // helpers would bind to the borrowing base and skip ownership registration.
  static void replaceOp(mlir::ConversionPatternRewriter &rewriter, SourceOp op,
                        mlir::Operation *replacement) {
    mlir::OpBuilder::InsertionGuard const guard(rewriter);
    rewriter.setInsertionPointAfter(replacement);
    replaceOp(rewriter, op, replacement->getResults());
  }

  template <typename Op, typename... Args>
  static Op replaceOpWithNewOp(mlir::ConversionPatternRewriter &rewriter,
                               SourceOp op, Args &&...args) {
    Op replacement =
        Op::create(rewriter, op->getLoc(), std::forward<Args>(args)...);
    replaceOp(rewriter, op, replacement.getOperation());
    return replacement;
  }
};

// Arrays and tuples share the runtime representation and construction protocol.
template <typename Op>
class ConvertArrayConstructionOp final : public OwnershipConversionPattern<Op> {
public:
  using OwnershipConversionPattern<Op>::OwnershipConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const handle = FunctionGetGlobalOp::create(
        rewriter, op.getLoc(), CFunctionType::get(rewriter.getContext()),
        "ffi.Array");
    mlir::Value const result = FunctionCallOp::create(
        rewriter, op.getLoc(),
        this->getTypeConverter()->convertType(op.getType()), handle,
        adaptor.getOperands());
    ObjectDecRefOp::create(rewriter, op.getLoc(), handle);
    this->replaceOp(rewriter, op, {result});
    return mlir::success();
  }
};

class ConvertArrayGetOp final : public OwnershipConversionPattern<ArrayGetOp> {
public:
  using OwnershipConversionPattern::OwnershipConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const index =
        ToOp::create(rewriter, op.getLoc(), IntType::get(rewriter.getContext()),
                     adaptor.getIndex());
    mlir::Value const handle = FunctionGetGlobalOp::create(
        rewriter, op.getLoc(), CFunctionType::get(rewriter.getContext()),
        "ffi.ArrayGetItem");
    mlir::Value const result = FunctionCallOp::create(
        rewriter, op.getLoc(), getTypeConverter()->convertType(op.getType()),
        handle, mlir::ValueRange{adaptor.getContainer(), index});
    ObjectDecRefOp::create(rewriter, op.getLoc(), handle);
    this->replaceOp(rewriter, op, {result});
    return mlir::success();
  }
};

class ConvertArraySizeOp final
    : public ReplacementConversionPattern<ArraySizeOp> {
public:
  using ReplacementConversionPattern::ReplacementConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArraySizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const handle = FunctionGetGlobalOp::create(
        rewriter, op.getLoc(), CFunctionType::get(rewriter.getContext()),
        "ffi.ArraySize");
    mlir::Value const result = FunctionCallOp::create(
        rewriter, op.getLoc(), getTypeConverter()->convertType(op.getType()),
        handle, adaptor.getContainer());
    ObjectDecRefOp::create(rewriter, op.getLoc(), handle);
    this->replaceOp(rewriter, op, {result});
    return mlir::success();
  }
};

class ConvertConstantStrOp final
    : public OwnershipConversionPattern<ConstantStrOp> {
public:
  using OwnershipConversionPattern::OwnershipConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConstantStrOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const raw =
        ConstantRawStrOp::create(rewriter, op.getLoc(), op.getValueAttr());
    this->template replaceOpWithNewOp<UpcastOp>(
        rewriter, op, getTypeConverter()->convertType(op.getType()), raw);
    return mlir::success();
  }
};

class ConvertDowncastOp final
    : public ReplacementConversionPattern<DowncastOp> {
public:
  using ReplacementConversionPattern::ReplacementConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DowncastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type const target = getTypeConverter()->convertType(op.getType());
    // TODO: Preserve finite Union/Str type constraints before erasing them to
    // CAny and materialize runtime type checks.
    if (adaptor.getValue().getType() == target) {
      replaceOp(rewriter, op, adaptor.getValue());
    } else {
      replaceOpWithNewOp<DowncastOp>(rewriter, op, mlir::TypeRange{target},
                                     mlir::ValueRange{adaptor.getValue()},
                                     op->getAttrs());
    }
    return mlir::success();
  }
};

// Exports retain each original Opaque operand, including repeated occurrences.
// Keep this boundary rule separate from the ownership of producer results.
template <typename Op>
class ConvertExportOp final : public ReplacementConversionPattern<Op> {
public:
  ConvertExportOp(const mlir::TypeConverter &converter,
                  mlir::MLIRContext *context)
      : ReplacementConversionPattern<Op>(converter, context, 2) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    for (const mlir::OpOperand &operand : op->getOpOperands()) {
      if (mlir::isa<TVMFFICOpaqueABIType>(operand.get().getType())) {
        ObjectIncRefOp::create(
            rewriter, op->getLoc(),
            adaptor.getOperands()[operand.getOperandNumber()]);
      }
    }
    this->template replaceOpWithNewOp<Op>(
        rewriter, op, mlir::TypeRange{}, adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

// Only explicitly registered regionless operations use this representation-only
// rewrite. Producer registrations select the ownership-aware base explicitly.
template <typename Op,
          template <typename> class Base = ReplacementConversionPattern>
class ConvertGenericOp final : public Base<Op> {
public:
  ConvertGenericOp(const mlir::TypeConverter &converter,
                   mlir::MLIRContext *context)
      : Base<Op>(converter, context, 2) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(this->getTypeConverter()->convertTypes(
            op->getResultTypes(), types))) {
      return mlir::failure();
    }
    this->template replaceOpWithNewOp<Op>(
        rewriter, op, types, adaptor.getOperands(), op->getAttrs());
    return mlir::success();
  }
};

// Move branch regions without cloning their producers or cleanup registrations.
class ConvertIfOp final : public OwnershipConversionPattern<mlir::scf::IfOp> {
public:
  ConvertIfOp(const mlir::TypeConverter &converter, mlir::MLIRContext *context)
      : OwnershipConversionPattern(converter, context, 2) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types))) {
      return mlir::failure();
    }
    mlir::scf::IfOp replacement = mlir::scf::IfOp::create(
        rewriter, op.getLoc(), types, adaptor.getCondition(), false);
    replacement->setAttrs(op->getAttrs());
    rewriter.eraseBlock(replacement.thenBlock());
    rewriter.inlineRegionBefore(op.getThenRegion(), replacement.getThenRegion(),
                                replacement.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), replacement.getElseRegion(),
                                replacement.getElseRegion().end());
    this->replaceOp(rewriter, op, replacement.getOperation());
    return mlir::success();
  }
};

class ConvertTupleUnconstructOp final
    : public OwnershipConversionPattern<TupleUnconstructOp> {
public:
  using OwnershipConversionPattern::OwnershipConversionPattern;

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
          mlir::Value const handle = FunctionGetGlobalOp::create(
              rewriter, op.getLoc(), CFunctionType::get(rewriter.getContext()),
              "ffi.ArrayGetItem");
          mlir::Value const result = FunctionCallOp::create(
              rewriter, op.getLoc(), getTypeConverter()->convertType(type),
              handle, mlir::ValueRange{adaptor.getContainer(), boxed});
          ObjectDecRefOp::create(rewriter, op.getLoc(), handle);
          return result;
        });
    this->replaceOp(rewriter, op, results);
    return mlir::success();
  }
};

class DecomposeTVMFFIPass final
    : public impl::DecomposeTVMFFIBase<DecomposeTVMFFIPass> {
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    if (mlir::failed(validateOwnershipScopes(op))) {
      signalPassFailure();
      return;
    }
    mlir::MLIRContext *context = op->getContext();
    DecomposeTVMFFITypeConverter const converter(context);
    // TODO: Preserve finite Union/Str alternatives across type erasure for
    // packed ABI argument validation and success-only return checks.
    mlir::RewritePatternSet conversions(context);
    mlir::ConversionTarget target(*context);
    // Boundary patterns use benefit 2 to precede structural patterns (benefit
    // 1).
    conversions.add<
        ConvertArrayConstructionOp<ArrayCreateOp>,
        ConvertArrayConstructionOp<TupleConstructOp>, ConvertArrayGetOp,
        ConvertArraySizeOp, ConvertConstantStrOp, ConvertDowncastOp,
        ConvertExportOp<ReturnOp>, ConvertExportOp<mlir::func::ReturnOp>,
        ConvertExportOp<mlir::scf::YieldOp>,
        ConvertGenericOp<CallOp, OwnershipConversionPattern>,
        ConvertGenericOp<EqOp>,
        ConvertGenericOp<ExceptionOp, OwnershipConversionPattern>,
        ConvertGenericOp<FunctionCallOp, OwnershipConversionPattern>,
        ConvertGenericOp<FunctionGetGlobalOp, OwnershipConversionPattern>,
        ConvertGenericOp<GetOp>, ConvertGenericOp<MonadFailOp>,
        ConvertGenericOp<MonadPureOp>, ConvertGenericOp<MonadUnpackOp>,
        ConvertGenericOp<ObjectDecRefOp>, ConvertGenericOp<ObjectIncRefOp>,
        ConvertGenericOp<TensorCloneOp, OwnershipConversionPattern>,
        ConvertGenericOp<TensorCopyOp>, ConvertGenericOp<TensorDTypeOp>,
        ConvertGenericOp<TensorDeviceOp>, ConvertGenericOp<TensorDimOp>,
        ConvertGenericOp<TensorLiteralOp, OwnershipConversionPattern>,
        ConvertGenericOp<TensorSizeOp>, ConvertGenericOp<TensorStorageOffsetOp>,
        ConvertGenericOp<TensorStrideOp>, ConvertGenericOp<UpcastOp>,
        ConvertGenericOp<mlir::func::CallOp, OwnershipConversionPattern>,
        ConvertIfOp, ConvertTupleUnconstructOp>(converter, context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(conversions,
                                                                   converter);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        conversions, converter);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, conversions, target);
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(
        converter, conversions, target);
    auto isLegal = [&](mlir::Operation *operation) -> bool {
      return converter.isLegal(operation) &&
             llvm::all_of(operation->getRegions(),
                          [&](mlir::Region &region) -> bool {
                            return converter.isLegal(&region);
                          });
    };
    target.addIllegalOp<ArrayCreateOp, ArrayGetOp, ArraySizeOp, ConstantStrOp,
                        TupleConstructOp, TupleUnconstructOp>();
    target.markUnknownOpDynamicallyLegal(isLegal);
    // Apply the same representation legality to SCF boundaries.
    target.addDynamicallyLegalOp<mlir::scf::IfOp, mlir::scf::YieldOp>(isLegal);
    target.addDynamicallyLegalOp<FuncOp, mlir::func::FuncOp>(
        [&](mlir::Operation *operation) -> bool {
          mlir::FunctionOpInterface function =
              mlir::cast<mlir::FunctionOpInterface>(operation);
          mlir::Region &body = function.getFunctionBody();
          return converter.isLegal(function.getFunctionType()) &&
                 (body.empty() ||
                  converter.isLegal(body.front().getArgumentTypes()));
        });

    // Materialize ownership before folding can erase a producer or branch.
    // Rollback mode preserves original operand types until commit.
    mlir::ConversionConfig config;
    config.allowPatternRollback = true;
    config.foldingMode = mlir::DialectConversionFoldingMode::AfterPatterns;
    if (mlir::failed(mlir::applyPartialConversion(
            op, target, std::move(conversions), config))) {
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
} // namespace trident::tvm_ffi
