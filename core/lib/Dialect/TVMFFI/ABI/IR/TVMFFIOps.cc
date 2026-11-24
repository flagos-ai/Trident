//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/Support/TVMFFITypes.h"
#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <string>

namespace trident::tvm_ffi::abi {

mlir::Block *DeferOp::addEntryBlock(mlir::OpBuilder &builder) {
  if (mlir::Block *entry = getEntryBlock()) {
    return entry;
  }
  return builder.createBlock(&getBody());
}

mlir::Block *DeferOp::getEntryBlock() {
  mlir::Region &body = getBody();
  return body.empty() ? nullptr : &body.front();
}

mlir::LogicalResult DeferOp::verify() {
  mlir::Block &body = *getEntryBlock();
  if (body.getNumArguments() != 0) {
    return emitOpError("body must have no arguments");
  }
  if (body.empty() || !mlir::isa<DeferYieldOp>(body.back())) {
    return emitOpError("body must end with tvm_ffi_abi.defer_yield");
  }
  for (mlir::Operation &operation : body.without_terminator()) {
    if (operation.getNumRegions() != 0 ||
        operation.hasTrait<mlir::OpTrait::IsTerminator>()) {
      return emitOpError(
          "body must contain only operations without regions or terminators");
    }
  }
  return mlir::success();
}

mlir::LogicalResult DeferYieldOp::verify() {
  return mlir::isa<DeferOp>((*this)->getParentOp())
             ? mlir::success()
             : emitOpError("must terminate tvm_ffi_abi.defer");
}

mlir::OpFoldResult DowncastOp::fold(FoldAdaptor) {
  if (UpcastOp upcast = getValue().getDefiningOp<UpcastOp>();
      upcast && upcast.getValue().getType() == getType()) {
    return upcast.getValue();
  } else if (getValue().getType() == getType()) {
    return getValue();
  } else {
    return {};
  }
}

mlir::OpFoldResult GetOp::fold(FoldAdaptor) {
  if (ToOp to = getOperand().getDefiningOp<ToOp>()) {
    return to.getValue();
  } else if (ConstantBoolOp constant =
                 getOperand().getDefiningOp<ConstantBoolOp>()) {
    return constant.getValueAttr();
  } else if (ConstantFloatOp constant =
                 getOperand().getDefiningOp<ConstantFloatOp>()) {
    return constant.getValueAttr();
  } else if (ConstantIntOp constant =
                 getOperand().getDefiningOp<ConstantIntOp>()) {
    return constant.getValueAttr();
  } else {
    return {};
  }
}

mlir::OpFoldResult ToOp::fold(FoldAdaptor) {
  GetOp get = getValue().getDefiningOp<GetOp>();
  return get && get.getOperand().getType() == getType() ? get.getOperand()
                                                        : mlir::OpFoldResult{};
}

mlir::OpFoldResult UpcastOp::fold(FoldAdaptor) {
  return getValue().getType() == getType() ? getValue() : mlir::OpFoldResult{};
}

bool CallOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return lhs.size() == 2 && rhs.size() == 2 && lhs[0] == rhs[0] &&
         mlir::isa<AnyType>(lhs[1]) && mlir::isa<AnyType>(rhs[1]);
}

bool PackCallOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                         mlir::TypeRange rhs) {
  return CallOp::isCompatibleReturnTypes(lhs, rhs);
}

bool PackGetOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                        mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 && mlir::isa<AnyType>(lhs[0]) &&
         mlir::isa<AnyType>(rhs[0]);
}

bool ExceptionOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                          mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         mlir::isa<ExceptionType>(lhs[0]) && mlir::isa<ExceptionType>(rhs[0]);
}

bool FunctionGetGlobalOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                                  mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         mlir::isa<FunctionType>(lhs[0]) && mlir::isa<FunctionType>(rhs[0]);
}

bool TensorCloneOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                            mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 && mlir::isa<TensorType>(lhs[0]) &&
         mlir::isa<TensorType>(rhs[0]);
}

bool TensorLiteralOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                              mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 && mlir::isa<TensorType>(lhs[0]) &&
         mlir::isa<TensorType>(rhs[0]);
}

mlir::LogicalResult AsOp::verify() {
  auto resultType = getResult().getType();
  return mlir::isa<dlpack::DLTensorType>(resultType)
             ? mlir::success()
             : emitOpError("unsupported object view type ") << resultType;
}

mlir::LogicalResult ConstantDTypeOp::verify() {
  mlir::ArrayAttr values = getValue();
  return values.size() == 3 &&
                 llvm::all_of(values,
                              [](mlir::Attribute value) -> bool {
                                return mlir::isa<mlir::IntegerAttr>(value);
                              })
             ? mlir::success()
             : emitOpError("dtype result requires [code, bits, lanes]");
}

mlir::Block *PackFuncOp::addEntryBlock(mlir::OpBuilder &builder) {
  if (mlir::Block *entry = getEntryBlock()) {
    return entry;
  }
  llvm::SmallVector<mlir::Location> const locations(getArgumentTypes().size(),
                                                    getLoc());
  return builder.createBlock(&getBody(), {}, getArgumentTypes(), locations);
}

mlir::Block *PackFuncOp::getEntryBlock() {
  mlir::Region &body = getBody();
  return body.empty() ? nullptr : &body.front();
}

mlir::LogicalResult PackFuncOp::verify() {
  if (getArgumentTypes().size() != 1 ||
      !mlir::isa<PackArgumentsType>(getArgumentTypes().front())) {
    return emitOpError("requires exactly one pack.arguments input");
  }
  if (getResultTypes().size() != 1 ||
      !mlir::isa<TVMFFICABIType>(getResultTypes().front())) {
    return emitOpError("function must return exactly one C ABI value");
  }
  return mlir::success();
}

mlir::LogicalResult PackCallOp::verify() {
  mlir::Operation *callee = mlir::SymbolTable::lookupNearestSymbolFrom(
      getOperation(), getCalleeAttr());
  if (!callee) {
    // Function conversion replaces packed definitions with their ABI symbols.
    std::string const name = "__tvm_ffi_" + getCallee().str();
    callee = mlir::SymbolTable::lookupNearestSymbolFrom(
        getOperation(), mlir::StringAttr::get(getContext(), name));
  }
  if (mlir::isa_and_nonnull<PackFuncOp, mlir::func::FuncOp>(callee)) {
    return mlir::success();
  }
  if (mlir::isa_and_nonnull<FuncOp>(callee)) {
    return mlir::success();
  }
  return emitOpError("callee must be a local packed entry");
}

mlir::Block *FuncOp::addEntryBlock(mlir::OpBuilder &builder) {
  if (mlir::Block *entry = getEntryBlock()) {
    return entry;
  }
  llvm::SmallVector<mlir::Location> const locations(getArgumentTypes().size(),
                                                    getLoc());
  return builder.createBlock(&getBody(), {}, getArgumentTypes(), locations);
}

mlir::Block *FuncOp::getEntryBlock() {
  mlir::Region &body = getBody();
  return body.empty() ? nullptr : &body.front();
}

mlir::LogicalResult FuncOp::verify() {
  if (llvm::any_of(getArgumentTypes(), [](mlir::Type type) -> bool {
        return mlir::isa<PackArgumentsType>(type);
      })) {
    return emitOpError("pack.arguments inputs require pack.func");
  }
  if (getResultTypes().size() != 1 ||
      !mlir::isa<TVMFFICABIType>(getResultTypes().front())) {
    return emitOpError("function must return exactly one C ABI value; use a "
                       "tuple for multiple values and None for no value");
  }
  return mlir::success();
}

mlir::LogicalResult GetOp::inferReturnTypes(
    mlir::MLIRContext *, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::PropertyRef,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }
  mlir::Type const inputType = operands.front().getType();
  auto nativeTypeInterface =
      mlir::dyn_cast<TVMFFINativeTypeInterface>(inputType);
  if (!nativeTypeInterface) {
    return mlir::failure();
  }
  inferredReturnTypes.push_back(nativeTypeInterface.getNativeType());
  return mlir::success();
}

mlir::LogicalResult GetOp::verify() {
  mlir::Type const tvmFFIType = getOperand().getType();
  mlir::Type const resultType = getResult().getType();
  auto nativeTypeInterface =
      mlir::dyn_cast<TVMFFINativeTypeInterface>(tvmFFIType);
  return nativeTypeInterface &&
                 nativeTypeInterface.getNativeType() == resultType
             ? mlir::success()
             : emitOpError("unsupported get from ")
                   << tvmFFIType << " to " << resultType;
}

mlir::LogicalResult NativeKernelLaunchOp::verify() {
  for (mlir::Type const type : getKernelOperands().getTypes()) {
    if (!mlir::isa<mlir::IntegerType, mlir::FloatType,
                   mlir::LLVM::LLVMPointerType>(type)) {
      return emitOpError(
          "kernel operands must have native integer, float or pointer types");
    }
  }
  return mlir::success();
}

mlir::LogicalResult CallOp::verify() {
  auto matches = [](mlir::Type actual, mlir::Type expected) -> bool {
    return actual == expected;
  };
  FuncOp function = mlir::SymbolTable::lookupNearestSymbolFrom<FuncOp>(
      getOperation(), getCalleeAttr());
  if (function) {
    if (function.getArgumentTypes().size() != getNumOperands() ||
        !llvm::all_of(llvm::zip(getOperandTypes(), function.getArgumentTypes()),
                      [&](auto pair) -> bool {
                        auto [actual, expected] = pair;
                        return matches(actual, expected);
                      })) {
      return emitOpError("operand types must match the callee arguments");
    }
  }
  // After function conversion the implementation has one output slot and an
  // i32 status.
  mlir::func::FuncOp lowered =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          getOperation(), getCalleeAttr());
  if (lowered) {
    mlir::FunctionType const signature = lowered.getFunctionType();
    if (signature.getNumInputs() != getNumOperands() + 1 ||
        !llvm::all_of(
            llvm::zip(getOperandTypes(), signature.getInputs().drop_back()),
            [&](auto pair) -> bool {
              auto [actual, expected] = pair;
              return matches(actual, expected);
            }) ||
        !mlir::isa<mlir::LLVM::LLVMPointerType>(signature.getInputs().back()) ||
        signature.getNumResults() != 1 ||
        !signature.getResult(0).isSignlessInteger(32)) {
      return emitOpError(
          "operand types must match the lowered callee arguments");
    }
  }
  return mlir::success();
}

mlir::LogicalResult IsaOp::verify() {
  mlir::Type const target = getTarget();
  if (!mlir::isa<TVMFFITypeIndexInterface>(target)) {
    return emitOpError("target must have a concrete runtime type index");
  }
  return mlir::success();
}

mlir::LogicalResult TensorLiteralOp::verify() {
  if (!mlir::isa<mlir::DenseElementsAttr>(getValue())) {
    return emitOpError("requires a dense elements attribute");
  }
  mlir::RankedTensorType const type =
      mlir::dyn_cast<mlir::RankedTensorType>(getValue().getType());
  if (!type || !type.hasStaticShape()) {
    return emitOpError("requires a statically shaped ranked tensor literal");
  }
  if (!mlir::isa<mlir::FloatType, mlir::IntegerType>(type.getElementType())) {
    return emitOpError(
        "requires an integer, boolean, or floating-point element type");
  }
  return mlir::success();
}

mlir::LogicalResult ToOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location>,
    mlir::ValueRange operands, mlir::DictionaryAttr, mlir::PropertyRef,
    mlir::RegionRange, llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (operands.size() != 1) {
    return mlir::failure();
  }
  mlir::Type const nativeType = operands.front().getType();
  mlir::Type const resultType =
      llvm::TypeSwitch<mlir::Type, mlir::Type>(nativeType)
          .Case<mlir::Float64Type>([context](mlir::Float64Type) -> mlir::Type {
            return FloatType::get(context);
          })
          .Case<mlir::IntegerType>(
              [context](mlir::IntegerType type) -> mlir::Type {
                if (type.getWidth() == 1) {
                  return BoolType::get(context);
                }
                if (type.getWidth() == 64) {
                  return IntType::get(context);
                }
                return {};
              })
          .Default([](mlir::Type) -> mlir::Type { return {}; });
  if (!resultType) {
    return mlir::failure();
  }
  inferredReturnTypes.push_back(resultType);
  return mlir::success();
}

mlir::LogicalResult ToOp::verify() {
  mlir::Type const nativeType = getValue().getType();
  mlir::Type const resultType = getResult().getType();
  auto nativeTypeInterface =
      mlir::dyn_cast<TVMFFINativeTypeInterface>(resultType);
  if (nativeTypeInterface &&
      nativeTypeInterface.getNativeType() == nativeType) {
    return mlir::success();
  }
  return emitOpError("unsupported native-to-TVM FFI conversion from ")
         << nativeType << " to " << resultType;
}

} // namespace trident::tvm_ffi::abi
