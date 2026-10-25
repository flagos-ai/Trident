//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/Transforms/LowerKernelLaunch.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace trident::tvm_ffi {

#define GEN_PASS_DEF_LOWERKERNELLAUNCH
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

/// Expand specialization contracts into checks and explicit Monad Results.
class LowerKernelLaunchOp final
    : public mlir::OpConversionPattern<tvm_ffi::KernelLaunchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::KernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::ArrayAttr const specializations = op.getSpecializationsAttr();
    auto materializeOperand =
        [&](tvm_ffi::SpecializationAttrInterface specialization,
            mlir::Value source) -> mlir::Value {
      mlir::Type const kind = specialization.getTargetType();
      return source.getType() == kind
                 ? source
                 : getTypeConverter()->materializeTargetConversion(
                       rewriter, loc, kind, source);
    };
    mlir::Value allChecks =
        mlir::arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    for (auto [specialization, source] : llvm::zip(
             specializations.getAsRange<tvm_ffi::SpecializationAttrInterface>(),
             op.getKernelOperands())) {
      // Constant checks compare ABI values directly, including semantic
      // strings.
      mlir::Value const operand =
          mlir::isa<tvm_ffi::ConstantSpecializationAttr>(specialization)
              ? source
              : materializeOperand(specialization, source);
      if (!operand) {
        return op.emitOpError("failed to materialize a native kernel operand");
      }
      if (mlir::Value const check =
              specialization.buildCheck(rewriter, loc, operand)) {
        allChecks =
            mlir::arith::AndIOp::create(rewriter, loc, allChecks, check);
      }
    }
    mlir::scf::IfOp branch = mlir::scf::IfOp::create(
        rewriter, loc, op->getResultTypes(), allChecks, true);
    rewriter.setInsertionPointToStart(branch.elseBlock());
    tvm_ffi::MonadFailOp rejected =
        tvm_ffi::MonadFailOp::create(rewriter, loc, op->getResultTypes());
    mlir::scf::YieldOp::create(rewriter, loc, rejected.getResults());
    rewriter.setInsertionPointToStart(branch.thenBlock());
    // TODO: Materialized native values can borrow storage from their source
    // objects and must not outlive or cross blocks independently of them.
    // Add a block-transfer check for borrowed values so future conversions
    // cannot accidentally reuse the guard materialization in this block.
    llvm::SmallVector<mlir::Value> operands;
    for (auto [specialization, source] : llvm::make_filter_range(
             llvm::zip(specializations
                           .getAsRange<tvm_ffi::SpecializationAttrInterface>(),
                       op.getKernelOperands()),
             [](auto it) -> bool {
               auto [specialization, source] = it;
               return !mlir::isa<tvm_ffi::ConstantSpecializationAttr>(
                   specialization);
             })) {
      mlir::Value const operand = materializeOperand(specialization, source);
      if (!operand) {
        return op.emitOpError("failed to materialize a native kernel operand");
      }
      operands.push_back(operand);
    }
    tvm_ffi::NativeKernelLaunchOp::create(
        rewriter, loc, op.getKernelAttr(), adaptor.getGridSizeX(),
        adaptor.getGridSizeY(), adaptor.getGridSizeZ(), adaptor.getBlockSizeX(),
        adaptor.getBlockSizeY(), adaptor.getBlockSizeZ(),
        adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
        adaptor.getClusterSizeZ(), adaptor.getDynamicSharedMemorySize(),
        operands);
    mlir::Value const none = tvm_ffi::ConstantNoneOp::create(rewriter, loc);
    tvm_ffi::MonadPureOp result = tvm_ffi::MonadPureOp::create(
        rewriter, loc, op->getResultTypes(), mlir::ValueRange{none});
    mlir::scf::YieldOp::create(rewriter, loc, result.getResults());
    rewriter.replaceOp(op, branch.getResults());
    return mlir::success();
  }
};

class LowerKernelLaunchPass final
    : public impl::LowerKernelLaunchBase<LowerKernelLaunchPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::TypeConverter typeConverter;
    typeConverter.addConversion(
        [](mlir::Type type) -> mlir::Type { return type; });
    typeConverter.addTargetMaterialization([](mlir::OpBuilder &builder,
                                              mlir::Type resultType,
                                              mlir::ValueRange inputs,
                                              mlir::Location loc)
                                               -> mlir::Value {
      if (inputs.size() != 1) {
        return {};
      }
      mlir::Value input = inputs.front();
      if (mlir::isa<tvm_ffi::TensorType, tvm_ffi::CTensorType>(
              input.getType())) {
        if (!mlir::isa<mlir::LLVM::LLVMPointerType>(resultType)) {
          return {};
        }
        mlir::Value const object = tvm_ffi::GetOp::create(
            builder, loc, tvm_ffi::ObjectType::get(builder.getContext()),
            input);
        mlir::Value const tensor = tvm_ffi::AsOp::create(
            builder, loc, dlpack::DLTensorType::get(builder.getContext()),
            object);
        return dlpack::TensorDataOp::create(builder, loc, resultType, tensor);
      }
      if (mlir::isa<tvm_ffi::BoolType, tvm_ffi::FloatType, tvm_ffi::IntType>(
              input.getType())) {
        input = tvm_ffi::GetOp::create(builder, loc, input);
      }
      mlir::Type const inputType = input.getType();
      if (inputType == resultType) {
        return input;
      }
      if (auto integer = mlir::dyn_cast<mlir::IntegerType>(inputType)) {
        auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(resultType);
        if (!targetInteger) {
          return {};
        }
        if (integer.getWidth() > targetInteger.getWidth()) {
          return mlir::arith::TruncIOp::create(builder, loc, resultType, input);
        }
        if (integer.isInteger(1)) {
          return mlir::arith::ExtUIOp::create(builder, loc, resultType, input);
        }
        return mlir::arith::ExtSIOp::create(builder, loc, resultType, input);
      }
      if (mlir::isa<mlir::FloatType>(inputType) &&
          mlir::isa<mlir::FloatType>(resultType)) {
        return mlir::arith::TruncFOp::create(builder, loc, resultType, input);
      }
      return {};
    });

    target.addIllegalOp<tvm_ffi::KernelLaunchOp>();
    target
        .addLegalDialect<dlpack::DLPackDialect, mlir::BuiltinDialect,
                         mlir::LLVM::LLVMDialect, mlir::arith::ArithDialect,
                         mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, tvm_ffi::TVMFFIDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    populateLowerKernelLaunchConversionPatterns(patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateLowerKernelLaunchConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter) {
  patterns.add<LowerKernelLaunchOp>(typeConverter, patterns.getContext());
}

} // namespace trident::tvm_ffi
