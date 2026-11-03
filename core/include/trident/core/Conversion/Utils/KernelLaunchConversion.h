//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_KERNELLAUNCHCONVERSION_H_
#define TRIDENT_CORE_CONVERSION_UTILS_KERNELLAUNCHCONVERSION_H_

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include <mlir/Transforms/DialectConversion.h>

namespace trident::conversion::utils {

void configureKernelLaunchTypeConverter(mlir::TypeConverter &typeConverter);

mlir::Value materializeKernelOperand(
    const mlir::TypeConverter &typeConverter,
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    tvm_ffi::SpecializationAttrInterface specialization, mlir::Value source);

mlir::LogicalResult buildKernelSpecializationGuard(
    tvm_ffi::KernelLaunchOp op, const mlir::TypeConverter &typeConverter,
    mlir::ConversionPatternRewriter &rewriter);

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_KERNELLAUNCHCONVERSION_H_
