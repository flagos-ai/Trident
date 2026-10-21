//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
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
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <optional>

namespace trident::tvm_ffi {

mlir::LogicalResult DeferOp::verify() {
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 0) {
    return emitOpError("body must have no arguments");
  }
  if (body.empty() || !mlir::isa<DeferYieldOp>(body.back())) {
    return emitOpError("body must end with tvm_ffi.defer_yield");
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
             : emitOpError("must terminate tvm_ffi.defer");
}

mlir::OpFoldResult DowncastOp::fold(FoldAdaptor) {
  return getValue().getType() == getType() ? getValue() : mlir::OpFoldResult{};
}

mlir::OpFoldResult UpcastOp::fold(FoldAdaptor) {
  return getValue().getType() == getType() ? getValue() : mlir::OpFoldResult{};
}

bool CallOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return lhs.size() == 2 && rhs.size() == 2 && lhs[0] == rhs[0] &&
         mlir::isa<AnyType, CAnyType>(lhs[1]) &&
         mlir::isa<AnyType, CAnyType>(rhs[1]);
}

bool ExceptionOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                          mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         mlir::isa<ExceptionType, CExceptionType>(lhs[0]) &&
         mlir::isa<ExceptionType, CExceptionType>(rhs[0]);
}

bool FunctionGetGlobalOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                                  mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         mlir::isa<FunctionType, CFunctionType>(lhs[0]) &&
         mlir::isa<FunctionType, CFunctionType>(rhs[0]);
}

bool TensorCloneOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                            mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         mlir::isa<TensorType, CTensorType>(lhs[0]) &&
         mlir::isa<TensorType, CTensorType>(rhs[0]);
}

bool TensorLiteralOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                              mlir::TypeRange rhs) {
  return lhs.size() == 1 && rhs.size() == 1 &&
         mlir::isa<TensorType, CTensorType>(lhs[0]) &&
         mlir::isa<TensorType, CTensorType>(rhs[0]);
}

mlir::LogicalResult ArrayCreateOp::verify() {
  ArrayType const type = getResult().getType();
  if (!llvm::all_of(getElements(), [&](mlir::Value value) -> bool {
        return value.getType() == type.getElementType();
      })) {
    return emitOpError("elements must match the array element type");
  }
  return mlir::success();
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

mlir::LogicalResult FuncOp::verify() {
  if (std::optional<llvm::StringRef> visibility = getSymVisibility();
      visibility && *visibility != "public") {
    return emitOpError("must have public visibility");
  } else if (getResultTypes().size() != 1 ||
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

mlir::ParseResult KernelLaunchOp::parseKernelArguments(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<mlir::Type> &types,
    mlir::ArrayAttr &specializations) {
  llvm::SmallVector<mlir::Attribute> parsedSpecializations;

  do {
    mlir::OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand) || parser.parseColon()) {
      return mlir::failure();
    }

    mlir::Type type;
    if (parser.parseType(type)) {
      return mlir::failure();
    }

    mlir::Attribute specialization;
    if (parser.parseAttribute(specialization)) {
      return mlir::failure();
    }
    parsedSpecializations.push_back(specialization);
    operands.push_back(operand);
    types.push_back(type);
  } while (mlir::succeeded(parser.parseOptionalComma()));

  specializations =
      mlir::ArrayAttr::get(parser.getContext(), parsedSpecializations);
  return mlir::success();
}

void KernelLaunchOp::printKernelArguments(mlir::OpAsmPrinter &printer,
                                          mlir::Operation *op [[maybe_unused]],
                                          mlir::OperandRange operands,
                                          mlir::TypeRange types,
                                          mlir::ArrayAttr specializations) {
  assert(operands.size() == types.size());
  assert(specializations.size() == operands.size());

  llvm::interleaveComma(llvm::enumerate(llvm::zip(operands, types)),
                        printer.getStream(), [&](auto indexedOperandAndType) {
                          auto [index, operandsAndType] = indexedOperandAndType;
                          auto [operand, type] = operandsAndType;
                          printer.printOperand(operand);
                          printer << " : ";
                          printer.printType(type);
                          printer << ' ';
                          printer.printAttribute(specializations[index]);
                        });
}

mlir::LogicalResult KernelLaunchOp::verify() {
  if (!mlir::isa<NoneType>(
          mlir::cast<ResultType>(getResult().getType()).getPayloadType())) {
    return emitOpError("must return Result<None>");
  }
  if (getSpecializations().size() != getKernelOperands().size()) {
    return emitOpError(
        "specializations and kernel operands must have the same size");
  }
  return mlir::success();
}

mlir::LogicalResult CallOp::verify() {
  // After function conversion the implementation has one output slot and an
  // i32 status. Preserve the ABI marker so this boundary remains verifiable.
  mlir::func::FuncOp lowered =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          getOperation(), getCalleeAttr());
  if (lowered) {
    mlir::FunctionType const signature = lowered.getFunctionType();
    if (!lowered->hasAttr("emit_tvm_ffi_abi")) {
      return emitOpError("callee must have emit_tvm_ffi_abi");
    }
    if (signature.getNumInputs() != getNumOperands() + 1 ||
        !llvm::equal(signature.getInputs().drop_back(), getOperandTypes()) ||
        !mlir::isa<mlir::LLVM::LLVMPointerType>(signature.getInputs().back()) ||
        signature.getNumResults() != 1 ||
        !signature.getResult(0).isSignlessInteger(32)) {
      return emitOpError(
          "operand types must match the lowered callee arguments");
    }
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

} // namespace trident::tvm_ffi
