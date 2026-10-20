// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/DecomposeTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
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
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/AttrTypeSubElements.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinTypes.h>
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
    if (mlir::Attribute const cleanup = op->getAttr(kCleanupAttributeName)) {
      call->setAttr(kCleanupAttributeName, cleanup);
    }
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
    if (mlir::Attribute const cleanup = op->getAttr(kCleanupAttributeName)) {
      call->setAttr(kCleanupAttributeName, cleanup);
    }
    rewriter.setInsertionPointAfter(call);
    ObjectDecRefOp::create(rewriter, call.getLoc(), handle);
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
    if (mlir::Attribute const cleanup = op->getAttr(kCleanupAttributeName)) {
      call->setAttr(kCleanupAttributeName, cleanup);
    }
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
      TVMFFICABIType const payload = mlir::dyn_cast_if_present<TVMFFICABIType>(
          convertType(type.getPayloadType()));
      return payload ? ResultType::get(context, payload) : mlir::Type{};
    });
    addConversion([context](TVMFFICOpaqueABIType type) -> mlir::Type {
      // Containers erase to the concrete Array ABI. Other opaque types retain
      // their specialized ABI/LLVM lowering paths.
      return mlir::isa<ArrayType, TupleType>(type)
                 ? mlir::Type(CArrayType::get(context))
                 : type;
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
          if (mlir::Attribute const cleanup =
                  op->getAttr(kCleanupAttributeName)) {
            value.getDefiningOp()->setAttr(kCleanupAttributeName, cleanup);
          }
          ObjectDecRefOp::create(rewriter, op.getLoc(), handle);
          return value;
        });
    rewriter.replaceOp(op, elements);
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
    target.addIllegalOp<ArrayCreateOp, ArrayGetOp, ArraySizeOp,
                        TupleConstructOp, TupleUnconstructOp>();
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
    conversions.add<DecomposeArrayGetOp, DecomposeArraySizeOp,
                    DecomposeContainerCreateOp<ArrayCreateOp>,
                    DecomposeContainerCreateOp<TupleConstructOp>,
                    DecomposeTupleUnconstructOp>(converter, op->getContext());
    conversions
        .add<GenericOpPattern<mlir::func::ReturnOp>, GenericOpPattern<CallOp>,
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
    conversions.add<GenericOpPattern<mlir::func::CallOp>>(converter,
                                                          op->getContext());
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
