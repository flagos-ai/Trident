// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/Transforms/LowerTVMFFIMonad.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <array>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/SCF/IR/SCF.h>         // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace trident::tvm_ffi {
#define GEN_PASS_DEF_LOWERTVMFFIMONAD
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"
namespace {

class FuncConversion final : public mlir::OpConversionPattern<tvm_ffi::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::FuncOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypeConverter const &converter = *getTypeConverter();
    mlir::TypeConverter::SignatureConversion arguments(op.getNumArguments());
    if (mlir::failed(
            converter.convertSignatureArgs(op.getArgumentTypes(), arguments))) {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Type> const outputs =
        llvm::SmallVector<mlir::Type>(op.getResultTypes());
    mlir::ArrayAttr argAttrs;
    if (mlir::ArrayAttr const attrs = op.getArgAttrsAttr()) {
      llvm::SmallVector<mlir::Attribute> converted;
      for (auto [type, attr] : llvm::zip_equal(op.getArgumentTypes(), attrs)) {
        converted.push_back(attr);
        if (mlir::isa<tvm_ffi::ResultType>(type)) {
          converted.push_back(rewriter.getDictionaryAttr({}));
        }
      }
      argAttrs = rewriter.getArrayAttr(converted);
    }
    if (!op.isExternal()) {
      rewriter.applySignatureConversion(&op.getBody().front(), arguments,
                                        &converter);
    }
    rewriter.modifyOpInPlace(op, [&] {
      op.setType(
          rewriter.getFunctionType(arguments.getConvertedTypes(), outputs));
      if (argAttrs) {
        op.setArgAttrsAttr(argAttrs);
      }
    });
    return mlir::success();
  }
};

template <typename SourceOp, bool successful>
class ResultConstructionConversion final
    : public mlir::OpConversionPattern<SourceOp> {
public:
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;
  mlir::LogicalResult matchAndRewrite(
      SourceOp op,
      typename mlir::OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const success = mlir::arith::ConstantIntOp::create(
        rewriter, op.getLoc(), successful, 1);
    llvm::SmallVector<std::array<mlir::Value, 2>> const replacements =
        llvm::map_to_vector(
            llvm::enumerate(op.getOutputs()),
            [&](const auto &entry) -> std::array<mlir::Value, 2> {
              auto [index, output] = entry;
              mlir::Value payload;
              if constexpr (successful) {
                payload = adaptor.getValues()[index].front();
              } else {
                mlir::Type const type =
                    mlir::cast<tvm_ffi::ResultType>(output.getType())
                        .getPayloadType();
                tvm_ffi::TVMFFIABIType const abi =
                    mlir::dyn_cast<tvm_ffi::TVMFFIABIType>(type);
                payload = abi ? tvm_ffi::TVMFFIABIType::getZero(
                                    rewriter, op.getLoc(), abi)
                              : mlir::ub::PoisonOp::create(rewriter,
                                                           op.getLoc(), type);
              }
              return {payload, success};
            });
    rewriter.replaceOpWithMultiple(
        op, llvm::map_to_vector(replacements,
                                [](const std::array<mlir::Value, 2> &values)
                                    -> mlir::ValueRange { return values; }));
    return mlir::success();
  }
};

class ReturnConversion final
    : public mlir::OpConversionPattern<tvm_ffi::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ReturnOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> const values = llvm::map_to_vector(
        llvm::zip_equal(op.getOperands(), adaptor.getOperands()),
        [&](const auto &entry) -> mlir::Value {
          auto [source, fields] = entry;
          return mlir::UnrealizedConversionCastOp::create(
                     rewriter, op.getLoc(), source.getType(), fields)
              .getResult(0);
        });
    rewriter.replaceOpWithNewOp<tvm_ffi::ReturnOp>(op, values);
    return mlir::success();
  }
};

class UnpackConversion final
    : public mlir::OpConversionPattern<tvm_ffi::MonadUnpackOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::MonadUnpackOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange const fields = adaptor.getInput();
    mlir::Value const payload = fields.front();
    mlir::Value const success = fields.back();
    if (!mlir::isa<tvm_ffi::AnyType>(op.getValue().getType())) {
      rewriter.replaceOp(op, {success, payload});
      return mlir::success();
    }
    mlir::Location const loc = op.getLoc();
    mlir::Type const any = op.getValue().getType();
    mlir::scf::IfOp branch = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{any}, success, true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    mlir::Value const widened =
        tvm_ffi::UpcastOp::create(rewriter, loc, any, payload);
    mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{widened});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    mlir::Value const none = tvm_ffi::ConstantNoneOp::create(rewriter, loc);
    mlir::Value const failure =
        tvm_ffi::UpcastOp::create(rewriter, loc, any, none);
    mlir::scf::YieldOp::create(rewriter, loc, mlir::ValueRange{failure});
    rewriter.replaceOp(op, {success, branch.getResult(0)});
    return mlir::success();
  }
};

class LowerTVMFFIMonadPass final
    : public impl::LowerTVMFFIMonadBase<LowerTVMFFIMonadPass> {
  void runOnOperation() override {
    // TODO: Refactor Result lowering, ownership analysis and ABI conversion to
    // preserve argument and result counts across lowering where possible, using
    // a one-to-one aggregate representation instead of expanding SSA values.
    mlir::TypeConverter converter;
    converter.addConversion([](mlir::Type type) -> mlir::Type { return type; });
    converter.addConversion(
        [&](tvm_ffi::ResultType type,
            llvm::SmallVectorImpl<mlir::Type> &types) -> mlir::LogicalResult {
          types.push_back(type.getPayloadType());
          types.push_back(mlir::IntegerType::get(&getContext(), 1));
          return mlir::success();
        });
    // Cross-function Result conversion is deliberately outside this pass.
    mlir::WalkResult const inlined = getOperation().walk(
        [&](mlir::func::FuncOp function) -> mlir::WalkResult {
          bool const hasResults =
              !converter.isLegal(function.getArgumentTypes()) ||
              !converter.isLegal(function.getResultTypes()) ||
              function
                  .walk([&](mlir::Operation *op) -> mlir::WalkResult {
                    return converter.isLegal(op)
                               ? mlir::WalkResult::advance()
                               : mlir::WalkResult::interrupt();
                  })
                  .wasInterrupted();
          if (hasResults) {
            function.emitError("inline ordinary functions carrying Results "
                               "before running lower-tvm-ffi-monad");
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (inlined.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FuncConversion,
                 ResultConstructionConversion<tvm_ffi::MonadFailOp, false>,
                 ResultConstructionConversion<tvm_ffi::MonadPureOp, true>,
                 ReturnConversion, UnpackConversion>(converter, &getContext());
    mlir::ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });
    // Only the return boundary retains Results; these casts are compile-time
    // materializations consumed by TVMFFIToFunc, with no runtime allocation.
    target.addLegalOp<tvm_ffi::ReturnOp, mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<tvm_ffi::ReturnOp>(
        [&](tvm_ffi::ReturnOp returned) -> bool {
          return llvm::all_of(
              returned.getOperands(), [&](mlir::Value value) -> bool {
                mlir::UnrealizedConversionCastOp fields =
                    value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
                return fields && fields->getBlock() == returned->getBlock() &&
                       fields.getNumOperands() == 2 &&
                       fields.getInputs().back().getType().isInteger(1) &&
                       converter.isLegal(fields.getOperandTypes());
              });
        });
    target.addIllegalOp<tvm_ffi::MonadBindOp, tvm_ffi::MonadFailOp,
                        tvm_ffi::MonadPureOp, tvm_ffi::MonadUnpackOp,
                        tvm_ffi::MonadYieldOp>();
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        converter, patterns, target);
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(converter,
                                                             patterns, target);
    target.addDynamicallyLegalOp<tvm_ffi::FuncOp>([&](tvm_ffi::FuncOp function)
                                                      -> bool {
      // The function pattern converts the entry block; CF patterns convert
      // successor blocks after the function itself has been legalized.
      return converter.isLegal(function.getArgumentTypes()) &&
             (function.isExternal() ||
              converter.isLegal(function.getBody().front().getArgumentTypes()));
    });
    llvm::SmallVector<tvm_ffi::FuncOp> functions;
    mlir::FrozenRewritePatternSet const frozenPatterns(std::move(patterns));
    getOperation().walk(
        [&](tvm_ffi::FuncOp function) { functions.push_back(function); });
    for (tvm_ffi::FuncOp const function : functions) {
      if (mlir::failed(
              mlir::applyPartialConversion(function, target, frozenPatterns))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace
} // namespace trident::tvm_ffi
