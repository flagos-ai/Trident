//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToFunc/TVMFFIToFunc.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>
#include <utility>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

class ConvertPackFuncOp final
    : public mlir::OpConversionPattern<tvm_ffi::PackFuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::PackFuncOp packed, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = packed.getLoc();
    mlir::Type const ptr =
        mlir::LLVM::LLVMPointerType::get(packed.getContext());
    mlir::Type const i32 = rewriter.getI32Type();
    mlir::Type const view =
        mlir::LLVM::LLVMStructType::getLiteral(packed.getContext(), {ptr, i32});
    std::string const name = "__tvm_ffi_" + packed.getSymName().str();
    mlir::FunctionType const type =
        rewriter.getFunctionType({ptr, ptr, i32, ptr}, {i32});
    mlir::func::FuncOp function =
        mlir::func::FuncOp::create(rewriter, loc, name, type);
    function.setSymVisibilityAttr(packed.getSymVisibilityAttr());
    if (packed.getBody().empty()) {
      rewriter.eraseOp(packed);
      return mlir::success();
    }
    mlir::Block *entry =
        rewriter.createBlock(&function.getBody(), {}, type.getInputs(),
                             llvm::SmallVector<mlir::Location>(4, loc));
    rewriter.setInsertionPointToStart(entry);
    mlir::Value args = mlir::LLVM::UndefOp::create(rewriter, loc, view);
    args = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, args, entry->getArgument(1), llvm::ArrayRef<int64_t>{0});
    args = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, args, entry->getArgument(2), llvm::ArrayRef<int64_t>{1});
    mlir::Value const borrowed =
        mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, tvm_ffi::PackArgumentsType::get(packed.getContext()),
            args)
            .getResult(0);
    mlir::Type const payloadType = packed.getResultTypes().front();
    mlir::TypeConverter::SignatureConversion signature(1);
    signature.remapInput(0, borrowed);
    mlir::Block *body =
        rewriter.applySignatureConversion(packed.getEntryBlock(), signature);
    mlir::cf::BranchOp::create(rewriter, loc, body);
    rewriter.inlineRegionBefore(packed.getBody(), function.getBody(),
                                function.getBody().end());
    for (mlir::Block &block : function.getBody()) {
      if (tvm_ffi::ReturnOp returned =
              mlir::dyn_cast<tvm_ffi::ReturnOp>(block.getTerminator())) {
        // A failed call leaves the caller's output slot untouched.
        mlir::Value const outputSlot = entry->getArgument(3);
        mlir::Location const loc = returned.getLoc();
        rewriter.setInsertionPoint(returned);
        mlir::Value const ready = tvm_ffi::GetOp::create(
            rewriter, loc, rewriter.getI1Type(), returned.getSuccess());
        mlir::scf::IfOp branch = mlir::scf::IfOp::create(
            rewriter, loc, mlir::TypeRange{rewriter.getI32Type()}, ready, true);
        rewriter.setInsertionPointToStart(branch.thenBlock());
        mlir::Type const anyType =
            tvm_ffi::TVMFFICABIType::getLLVMType(rewriter.getContext());
        mlir::Value const checked = rewriter.createOrFold<tvm_ffi::DowncastOp>(
            loc, payloadType, returned.getValue());
        mlir::Value const payload = rewriter.createOrFold<tvm_ffi::UpcastOp>(
            loc, tvm_ffi::CAnyType::get(rewriter.getContext()), checked);
        mlir::Value const value = mlir::UnrealizedConversionCastOp::create(
                                      rewriter, loc, anyType, payload)
                                      .getResult(0);
        mlir::LLVM::StoreOp::create(rewriter, loc, value, outputSlot);
        mlir::Value const zero =
            mlir::arith::ConstantIntOp::create(rewriter, loc, 0, 32);
        mlir::scf::YieldOp::create(rewriter, loc, zero);
        rewriter.setInsertionPointToStart(branch.elseBlock());
        mlir::Value const error =
            mlir::arith::ConstantIntOp::create(rewriter, loc, -1, 32);
        mlir::scf::YieldOp::create(rewriter, loc, error);
        rewriter.setInsertionPoint(returned);
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returned,
                                                          branch.getResults());
      }
    }
    rewriter.eraseOp(packed);
    return mlir::success();
  }
};

class ConvertTVMFFIToFuncPass final
    : public impl::ConvertTVMFFIToFuncBase<ConvertTVMFFIToFuncPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    populateTVMFFIToFuncConversionPatterns(target, patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateTVMFFIToFuncConversionPatterns(mlir::ConversionTarget &target,
                                            mlir::RewritePatternSet &patterns) {
  target.addIllegalOp<tvm_ffi::PackFuncOp>();
  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *) -> bool { return true; });
  patterns.add<ConvertPackFuncOp>(patterns.getContext());
}

} // namespace trident::conversion
