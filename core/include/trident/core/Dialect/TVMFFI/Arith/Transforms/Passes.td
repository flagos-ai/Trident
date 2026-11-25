// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "mlir/Pass/PassBase.td"
def LowerTVMFFIArith : Pass<"lower-tvm-ffi-arith", "mlir::ModuleOp"> {
  let summary = "Lower promoted TVM FFI scalar arithmetic to native arith";
  // clang-format off
  let description = [{
    Extract Bool/Int/Float operands, promote them to i1/i64/f64, compute using
    native arith operations and box results. Run before scalar canonicalization
    and LLVM lowering. Any and Union operands require explicit narrowing.
  }];
  // clang-format on
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "trident::tvm_ffi::abi::TVMFFIABIDialect",
                           "trident::tvm_ffi::arith::TVMFFIArithDialect"];
}
