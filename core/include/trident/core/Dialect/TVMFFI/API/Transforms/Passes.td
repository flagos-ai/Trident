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
def DirectAllocations : Pass<"direct-allocations", "mlir::ModuleOp"> {
  let summary =
      "Route supported ATen allocation and view ops through the direct C ABI";
  let dependentDialects = ["trident::tvm_ffi::api::TVMFFIAPIDialect",
                           "trident::tvm_ffi::monad::TVMFFIMonadDialect"];
}

def FlattenDirectAllocations
    : Pass<"flatten-direct-allocations", "mlir::ModuleOp"> {
  let summary =
      "Replace local shape arrays with native direct allocation dimensions";
  let dependentDialects = ["trident::tvm_ffi::api::TVMFFIAPIDialect"];
}

def UnboxScfIfTuples : Pass<"unbox-scf-if-tuples", "mlir::ModuleOp"> {
  let summary = "Unbox semantic tuple results across scf.if boundaries";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "trident::tvm_ffi::abi::TVMFFIABIDialect",
                           "trident::tvm_ffi::api::TVMFFIAPIDialect"];
}
