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
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <optional>

namespace trident::tvm_ffi {

namespace {

llvm::SmallVector<mlir::Type> getPayloadTypes(mlir::ValueRange values) {
  return llvm::map_to_vector(values, [](mlir::Value value) -> mlir::Type {
    return mlir::cast<ResultType>(value.getType()).getPayloadType();
  });
}

} // namespace

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
             !mlir::isa<ResultType>(getResultTypes().front())) {
    return emitOpError("function must return exactly one monadic Result; use a "
                       "tuple for multiple values and None for no value");
  } else {
    return mlir::success();
  }
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
  if (getSpecializations().size() != getKernelOperands().size()) {
    return emitOpError(
        "specializations and kernel operands must have the same size");
  }
  return mlir::success();
}

mlir::LogicalResult MonadBindOp::verify() {
  if (getInputs().empty() || getOutputs().empty()) {
    return emitOpError("requires at least one input and output Result");
  } else if (!getBody().hasOneBlock() || getBody().front().empty()) {
    return emitOpError(
        "body must contain one nonempty block ending with tvm_ffi.monad.yield");
  } else if (llvm::SmallVector<mlir::Type> const payloads =
                 getPayloadTypes(getInputs());
             getBody().front().getArgumentTypes() !=
             mlir::TypeRange(payloads)) {
    return emitOpError("region arguments must match input Result payloads");
  } else if (!mlir::isa<MonadYieldOp>(getBody().front().getTerminator())) {
    return emitOpError("body must end with tvm_ffi.monad.yield");
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult MonadFailOp::verify() {
  return getOutputs().empty() ? emitOpError("requires at least one Result")
                              : mlir::success();
}

mlir::LogicalResult MonadPureOp::verify() {
  llvm::SmallVector<mlir::Type> const payloads = getPayloadTypes(getOutputs());
  if (getOutputs().empty()) {
    return emitOpError("requires at least one Result");
  } else if (getValues().size() != payloads.size()) {
    return emitOpError("payload count must match Result count");
  } else if (!llvm::equal(getValues().getTypes(), payloads)) {
    return emitOpError("payload type must match Result type");
  } else {
    return mlir::success();
  }
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

mlir::LogicalResult ReturnOp::verify() {
  FuncOp function = mlir::cast<FuncOp>(getOperation()->getParentOp());
  if (getOperandTypes() != function.getResultTypes()) {
    return emitOpError("Result types must match the function results");
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
