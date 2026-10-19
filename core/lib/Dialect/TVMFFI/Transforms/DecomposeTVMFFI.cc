// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/DecomposeTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
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
#define GEN_PASS_DEF_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

namespace {

class DecomposeArrayGetOp final : public mlir::OpConversionPattern<ArrayGetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const index =
        ToOp::create(rewriter, op.getLoc(), IntType::get(rewriter.getContext()),
                     adaptor.getIndex());
    mlir::Type const resultType = getTypeConverter()->convertType(op.getType());
    mlir::Value const handle =
        FunctionGetGlobalOp::create(rewriter, op.getLoc(), "ffi.ArrayGetItem");

    FunctionCallOp call = rewriter.replaceOpWithNewOp<FunctionCallOp>(
        op, resultType, handle,
        mlir::ValueRange{adaptor.getContainer(), index});
    call->setAttr("tvm_ffi.cleanup", rewriter.getUnitAttr());
    rewriter.setInsertionPointAfter(call);
    ObjectDecRefOp::create(rewriter, call.getLoc(), handle);
    return mlir::success();
  }
};

class DecomposeArraySizeOp final
    : public mlir::OpConversionPattern<ArraySizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArraySizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const handle =
        FunctionGetGlobalOp::create(rewriter, op.getLoc(), "ffi.ArraySize");

    FunctionCallOp call = rewriter.replaceOpWithNewOp<FunctionCallOp>(
        op, op.getType(), handle, mlir::ValueRange{adaptor.getContainer()});
    call->setAttr("tvm_ffi.cleanup", rewriter.getUnitAttr());
    rewriter.setInsertionPointAfter(call);
    ObjectDecRefOp::create(rewriter, call.getLoc(), handle);
    return mlir::success();
  }
};

class DecomposeBindOp final : public mlir::OpConversionPattern<MonadBindOp> {
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
          MonadUnpackOp unpack = MonadUnpackOp::create(rewriter, loc, input);
          mlir::Value const success = unpack.getSuccess();
          ready =
              rewriter.createOrFold<mlir::arith::AndIOp>(loc, ready, success);
          return unpack.getValue();
        });
    // TODO: Eliminate the redundant Any selection and downcasts introduced by
    // unpacking typed Results when expanding this standard bind form.
    mlir::scf::IfOp branch = mlir::scf::IfOp::create(
        rewriter, loc, resultTypes, ready, !resultTypes.empty());
    if (resultTypes.empty()) {
      rewriter.eraseOp(branch.thenBlock()->getTerminator());
    }
    if (mlir::Attribute const cleanup = bind->getAttr("tvm_ffi.cleanup")) {
      branch->setAttr("tvm_ffi.cleanup", cleanup);
    }
    rewriter.setInsertionPointToStart(branch.thenBlock());
    mlir::Block &body = *bind.getEntryBlock();
    llvm::SmallVector<mlir::Value> const payloads = llvm::map_to_vector(
        llvm::zip_equal(unpacked, body.getArguments()),
        [&](const auto &entry) -> mlir::Value {
          auto [value, argument] = entry;
          if (value.getType() != argument.getType()) {
            return DowncastOp::create(rewriter, loc, argument.getType(), value);
          }
          return value;
        });
    // Move rather than clone the continuation, as in SCF structural conversion.
    // Its yield is converted separately using the remapped operands.
    rewriter.mergeBlocks(&body, branch.thenBlock(), payloads);
    if (!resultTypes.empty()) {
      rewriter.setInsertionPointToStart(branch.elseBlock());
      MonadFailOp failed = MonadFailOp::create(rewriter, loc, resultTypes);
      // Failed Results carry no references and request no export retains.
      mlir::scf::YieldOp::create(rewriter, loc, failed.getResults());
    }
    rewriter.replaceOp(bind, branch.getResults());
    return mlir::success();
  }
};

// Array and Tuple construction both lower to the runtime Array constructor.
template <typename OpTy>
class DecomposeContainerCreateOp final
    : public mlir::OpConversionPattern<OpTy> {
public:
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type const resultType =
        this->getTypeConverter()->convertType(op.getType());
    mlir::Value const handle =
        FunctionGetGlobalOp::create(rewriter, op.getLoc(), "ffi.Array");

    FunctionCallOp call = rewriter.replaceOpWithNewOp<FunctionCallOp>(
        op, resultType, handle, adaptor.getElements());
    call->setAttr("tvm_ffi.cleanup", rewriter.getUnitAttr());
    rewriter.setInsertionPointAfter(call);
    ObjectDecRefOp::create(rewriter, call.getLoc(), handle);
    return mlir::success();
  }
};

class DecomposeTVMFFITypeConverter final : public mlir::TypeConverter {
public:
  explicit DecomposeTVMFFITypeConverter(mlir::MLIRContext *context) {
    // Register the identity fallback first: specific conversions take priority.
    addConversion([](mlir::Type type) -> mlir::Type { return type; });
    addConversion([this, context](mlir::FunctionType type) -> mlir::Type {
      llvm::SmallVector<mlir::Type> inputs;
      llvm::SmallVector<mlir::Type> outputs;
      if (mlir::failed(convertTypes(type.getInputs(), inputs)) ||
          mlir::failed(convertTypes(type.getResults(), outputs))) {
        return {};
      }
      return mlir::FunctionType::get(context, inputs, outputs);
    });
    addConversion([this, context](ResultType type) -> mlir::Type {
      return ResultType::get(context, convertType(type.getPayloadType()));
    });
    addConversion([context](TVMFFICOpaqueABIType type) -> mlir::Type {
      // Containers erase to the concrete Array ABI. Other opaque types retain
      // their specialized ABI/LLVM lowering paths.
      if (mlir::isa<ArrayType, TupleType>(type)) {
        return CArrayType::get(context);
      }
      return type;
    });
  }
};

class DecomposeTupleUnconstructOp final
    : public mlir::OpConversionPattern<TupleUnconstructOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TupleUnconstructOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> const elements = llvm::map_to_vector(
        llvm::enumerate(op.getContainer().getType().getTypes()),
        [&](const auto &entry) -> mlir::Value {
          auto [index, type] = entry;
          mlir::Value const offset = mlir::arith::ConstantIntOp::create(
              rewriter, op.getLoc(), static_cast<int64_t>(index), 64);
          mlir::Value const boxed =
              ToOp::create(rewriter, op.getLoc(),
                           IntType::get(rewriter.getContext()), offset);
          mlir::Value const handle = FunctionGetGlobalOp::create(
              rewriter, op.getLoc(), "ffi.ArrayGetItem");
          mlir::Value const value = FunctionCallOp::create(
              rewriter, op.getLoc(), getTypeConverter()->convertType(type),
              handle, mlir::ValueRange{adaptor.getContainer(), boxed});

          value.getDefiningOp()->setAttr("tvm_ffi.cleanup",
                                         rewriter.getUnitAttr());
          ObjectDecRefOp::create(rewriter, op.getLoc(), handle);
          return value;
        });
    rewriter.replaceOp(op, elements);
    return mlir::success();
  }
};

class DecomposeYieldOp final : public mlir::OpConversionPattern<MonadYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MonadYieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
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
    mlir::FailureOr<mlir::Operation *> replacement = mlir::convertOpResultTypes(
        op, adaptor.getOperands(), *this->getTypeConverter(), rewriter);
    if (mlir::failed(replacement)) {
      return mlir::failure();
    }
    if constexpr (std::is_same_v<OpTy, FunctionCallOp>) {
      (*replacement)->setAttr("tvm_ffi.cleanup", rewriter.getUnitAttr());
    }
    rewriter.replaceOp(op, (*replacement)->getResults());
    return mlir::success();
  }
};

// Pattern and type-converter definitions must precede pass registration.
class DecomposeTVMFFIPass final
    : public impl::DecomposeTVMFFIBase<DecomposeTVMFFIPass> {
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    DecomposeTVMFFITypeConverter converter(op->getContext());
    mlir::ConversionTarget target(*op->getContext());
    target.addIllegalOp<ArrayCreateOp, ArrayGetOp, ArraySizeOp, MonadBindOp,
                        MonadYieldOp, TupleConstructOp, TupleUnconstructOp>();
    target.markUnknownOpDynamicallyLegal([&](mlir::Operation *op) -> bool {
      return converter.isLegal(op) &&
             llvm::all_of(op->getRegions(), [&](mlir::Region &region) -> bool {
               return converter.isLegal(&region);
             });
    });
    target.addDynamicallyLegalOp<FuncOp, mlir::func::FuncOp>(
        [&](mlir::Operation *op) -> bool {
          mlir::FunctionOpInterface function =
              mlir::cast<mlir::FunctionOpInterface>(op);
          mlir::Region &body = function.getFunctionBody();
          // Function patterns convert entries; CF patterns convert successors.
          return converter.isLegal(function.getFunctionType()) &&
                 (body.empty() ||
                  converter.isLegal(body.front().getArgumentTypes()));
        });
    mlir::RewritePatternSet conversions(op->getContext());
    conversions.add<DecomposeArrayGetOp, DecomposeArraySizeOp, DecomposeBindOp,
                    DecomposeContainerCreateOp<ArrayCreateOp>,
                    DecomposeContainerCreateOp<TupleConstructOp>,
                    DecomposeTupleUnconstructOp, DecomposeYieldOp>(
        converter, op->getContext());
    conversions
        .add<GenericOpPattern<mlir::func::ReturnOp>,
             GenericOpPattern<DowncastOp>, GenericOpPattern<FunctionCallOp>,
             GenericOpPattern<MonadFailOp>, GenericOpPattern<MonadPureOp>,
             GenericOpPattern<MonadUnpackOp>, GenericOpPattern<ReturnOp>,
             GenericOpPattern<UpcastOp>>(converter, op->getContext());
    // Prefer the attribute-preserving yield rewrite over SCF's structural
    // fallback, which rebuilds yields without their pending export retains.
    conversions.add<GenericOpPattern<mlir::scf::YieldOp>>(
        converter, op->getContext(), mlir::PatternBenefit(2));
    mlir::populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(conversions,
                                                                   converter);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        conversions, converter);
    mlir::populateCallOpTypeConversionPattern(conversions, converter);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, conversions, target);
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(
        converter, conversions, target);
    if (mlir::failed(
            mlir::applyPartialConversion(op, target, std::move(conversions)))) {
      signalPassFailure();
      return;
    }
    // Convert type-bearing attributes separately from operation structure.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](mlir::Type type) -> mlir::Type {
      return converter.convertType(type);
    });
    replacer.recursivelyReplaceElementsIn(op);
  }
};

} // namespace
} // namespace trident::tvm_ffi
