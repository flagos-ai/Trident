// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_ARITH_TVMFFIOPTRAITS_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_ARITH_TVMFFIOPTRAITS_H_

#include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFITypes.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>

namespace trident::tvm_ffi::arith::OpTrait {

/// Return Bool independently of the promoted computation type.
template <typename ConcreteOp>
class ArithBoolResult
    : public mlir::OpTrait::TraitBase<ConcreteOp, ArithBoolResult> {};

/// Implement InferTypeOpInterface using the operation's promotion trait.
template <typename ConcreteOp, template <typename> class PromotionTrait>
class ArithTypeInference
    : public mlir::OpTrait::TraitBase<ConcreteOp, PromotionTrait> {
public:
  static mlir::LogicalResult
  inferReturnTypes(mlir::MLIRContext *, std::optional<mlir::Location>,
                   mlir::ValueRange operands, mlir::DictionaryAttr,
                   mlir::PropertyRef, mlir::RegionRange,
                   llvm::SmallVectorImpl<mlir::Type> &types) {
    mlir::FailureOr<mlir::Type> const type =
        ConcreteOp::getArithType(operands.getTypes());
    if (mlir::failed(type)) {
      return mlir::failure();
    }
    if constexpr (ConcreteOp::template hasTrait<ArithBoolResult>()) {
      types.push_back(abi::BoolType::get(type->getContext()));
    } else {
      types.push_back(*type);
    }
    return mlir::success();
  }
};

/// Require Bool/Int operands, retaining Bool for Bool-only computation.
template <typename ConcreteOp>
class ArithBitwisePromotion
    : public ArithTypeInference<ConcreteOp, ArithBitwisePromotion> {
public:
  static mlir::FailureOr<mlir::Type> getArithType(mlir::TypeRange types) {
    if (types.empty() || !llvm::all_of(types, [](mlir::Type type) -> bool {
          return mlir::isa<TVMFFIIntegralType>(type);
        })) {
      return mlir::failure();
    }
    mlir::MLIRContext *context = types.front().getContext();
    if (llvm::all_of(types, [](mlir::Type type) -> bool {
          return mlir::isa<abi::BoolType>(type);
        })) {
      return abi::BoolType::get(context);
    } else {
      return abi::IntType::get(context);
    }
  }
};

/// Promote every numeric operand to Float.
template <typename ConcreteOp>
class ArithFloatPromotion
    : public ArithTypeInference<ConcreteOp, ArithFloatPromotion> {
public:
  static mlir::FailureOr<mlir::Type> getArithType(mlir::TypeRange types) {
    if (types.empty() || !llvm::all_of(types, [](mlir::Type type) -> bool {
          return mlir::isa<TVMFFINumericType>(type);
        })) {
      return mlir::failure();
    } else {
      return abi::FloatType::get(types.front().getContext());
    }
  }
};

/// Require Bool/Int operands and promote them to Int.
template <typename ConcreteOp>
class ArithIntegralPromotion
    : public ArithTypeInference<ConcreteOp, ArithIntegralPromotion> {
public:
  static mlir::FailureOr<mlir::Type> getArithType(mlir::TypeRange types) {
    if (types.empty() || !llvm::all_of(types, [](mlir::Type type) -> bool {
          return mlir::isa<TVMFFIIntegralType>(type);
        })) {
      return mlir::failure();
    } else {
      return abi::IntType::get(types.front().getContext());
    }
  }
};

/// Promote Bool to Int and mixed Float operands to Float.
template <typename ConcreteOp>
class ArithNumericPromotion
    : public ArithTypeInference<ConcreteOp, ArithNumericPromotion> {
public:
  static mlir::FailureOr<mlir::Type> getArithType(mlir::TypeRange types) {
    if (types.empty() || !llvm::all_of(types, [](mlir::Type type) -> bool {
          return mlir::isa<TVMFFINumericType>(type);
        })) {
      return mlir::failure();
    }
    mlir::MLIRContext *context = types.front().getContext();
    if (llvm::any_of(types, [](mlir::Type type) -> bool {
          return mlir::isa<abi::FloatType>(type);
        })) {
      return abi::FloatType::get(context);
    } else {
      return abi::IntType::get(context);
    }
  }
};

} // namespace trident::tvm_ffi::arith::OpTrait

#endif
