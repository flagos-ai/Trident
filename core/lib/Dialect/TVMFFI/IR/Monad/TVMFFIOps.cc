// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

namespace trident::tvm_ffi {

static llvm::SmallVector<mlir::Type> getPayloadTypes(mlir::ValueRange values) {
  return llvm::map_to_vector(values, [](mlir::Value value) -> mlir::Type {
    return mlir::cast<ResultType>(value.getType()).getPayloadType();
  });
}

mlir::Block *MonadBindOp::addEntryBlock(mlir::OpBuilder &builder) {
  if (mlir::Block *entry = getEntryBlock()) {
    return entry;
  }
  llvm::SmallVector<mlir::Type> const payloads = getPayloadTypes(getInputs());
  llvm::SmallVector<mlir::Location> const locations(payloads.size(), getLoc());
  return builder.createBlock(&getBody(), {}, payloads, locations);
}

mlir::Block *MonadBindOp::getEntryBlock() {
  mlir::Region &body = getBody();
  return body.empty() ? nullptr : &body.front();
}

mlir::LogicalResult MonadBindOp::canonicalize(MonadBindOp op,
                                              mlir::PatternRewriter &rewriter) {
  mlir::Block &body = *op.getEntryBlock();
  // Replacing block arguments and moving the body require a region rewrite.
  for (auto it : llvm::enumerate(op.getInputs())) {
    std::size_t index = it.index();
    mlir::Value input = it.value();
    if (MonadPureOp pure = input.getDefiningOp<MonadPureOp>()) {
      rewriter.replaceAllUsesWith(body.getArgument(index), pure.getValue());
      rewriter.modifyOpInPlace(op, [&body, index, op] {
        op->eraseOperand(index);
        body.eraseArgument(index);
      });
      return mlir::success();
    }
  }
  MonadYieldOp yield = mlir::cast<MonadYieldOp>(body.getTerminator());
  if (!op.getInputs().empty() || !op->getAttrs().empty() ||
      !yield->getAttrs().empty()) {
    return mlir::failure();
  }
  rewriter.inlineBlockBefore(&body, op);
  rewriter.replaceOp(op, yield.getValues());
  rewriter.eraseOp(yield);
  return mlir::success();
}

mlir::LogicalResult MonadBindOp::verify() {
  mlir::Block *entry = getEntryBlock();
  if (!getBody().hasOneBlock() || !entry || entry->empty()) {
    return emitOpError(
        "body must contain one nonempty block ending with tvm_ffi.monad.yield");
  } else if (llvm::SmallVector<mlir::Type> const payloads =
                 getPayloadTypes(getInputs());
             entry->getArgumentTypes() != mlir::TypeRange(payloads)) {
    return emitOpError("region arguments must match input Result payloads");
  } else if (!mlir::isa<MonadYieldOp>(entry->getTerminator())) {
    return emitOpError("body must end with tvm_ffi.monad.yield");
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult MonadCallOp::verify() {
  FuncOp function = mlir::SymbolTable::lookupNearestSymbolFrom<FuncOp>(
      getOperation(), getCalleeAttr());
  return function && (function.getResultTypes().size() != 1 ||
                      getResult().getType().getPayloadType() !=
                          function.getResultTypes().front())
             ? emitOpError("Result payload must match the callee result")
             : mlir::success();
}

mlir::LogicalResult MonadFailOp::verify() {
  return getOutputs().empty() ? emitOpError("requires at least one Result")
                              : mlir::success();
}

bool MonadUnpackOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                            mlir::TypeRange rhs) {
  return lhs == rhs ||
         (lhs.size() == 2 && rhs.size() == 2 && lhs[0] == rhs[0] &&
          mlir::isa<AnyType, CAnyType>(lhs[1]) &&
          mlir::isa<AnyType, CAnyType>(rhs[1]));
}

mlir::LogicalResult MonadUnpackOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::PropertyRef,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }
  ResultType const result =
      mlir::dyn_cast<ResultType>(operands.front().getType());
  if (!result) {
    return mlir::failure();
  }
  mlir::Type const payload = result.getPayloadType();
  mlir::Type const any = AnyType::get(context);
  inferredReturnTypes.push_back(mlir::IntegerType::get(context, 1));
  inferredReturnTypes.push_back(
      UpcastOp::areCastCompatible({payload}, {any}) ? any : payload);
  return mlir::success();
}

mlir::LogicalResult MonadYieldOp::verify() {
  mlir::Operation *parent = getOperation()->getParentOp();
  if (MonadBindOp bind = mlir::dyn_cast<MonadBindOp>(parent)) {
    if (getOperandTypes() != bind.getResultTypes()) {
      return emitOpError("Result types must match bind outputs");
    }
  } else {
    return emitOpError("must be nested directly in tvm_ffi.monad.bind");
  }
  return mlir::success();
}

} // namespace trident::tvm_ffi
