// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "mlir/Pass/PassBase.td"

def LowerTVMFFIAPIKernelLaunch
    : Pass<"lower-tvm-ffi-api-kernel-launch", "mlir::ModuleOp"> {
  let summary = "Expand launch specializations into guards and Monad Results";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect", "mlir::LLVM::LLVMDialect",
                           "mlir::scf::SCFDialect",
                           "trident::dlpack::DLPackDialect",
                           "trident::tvm_ffi::abi::TVMFFIABIDialect",
                           "trident::tvm_ffi::api::TVMFFIAPIDialect",
                           "trident::tvm_ffi::monad::TVMFFIMonadDialect"];
}
