// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "mlir/Pass/PassBase.td"
def LowerTVMFFIMonad : Pass<"lower-tvm-ffi-monad", "mlir::ModuleOp"> {
  let summary = "Lower Results to semantic Bool/Any tuples";
  // clang-format off
  let description = [{
    Lower monadic calls to separate Bool/Any results followed by explicit tuple
    construction, and expand monadic control flow, converting every Result to
    tuple<bool, any>. Run after kernel lowering and before container decomposition
    and cleanup. No monadic operations or Result types remain.
  }];
  // clang-format on
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "trident::tvm_ffi::abi::TVMFFIABIDialect",
                           "trident::tvm_ffi::monad::TVMFFIMonadDialect",
                           "trident::tvm_ffi::api::TVMFFIAPIDialect"];
}
