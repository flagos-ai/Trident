// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "mlir/Pass/PassBase.td"

def LowerTVMFFIABIDefer : Pass<"lower-tvm-ffi-abi-defer", "mlir::ModuleOp"> {
  let summary = "Expand deferred cleanup at dominance-scope exits";
  let description = [{
    Clone defer bodies at registration dominance-scope exits with unspecified
    relative order, preserving operation order within each body and running
    after existing export retains. Support SSACFG regions with block terminators,
    independent of enclosing operation or exit kind. Terminators without CFG
    successors end the region execution; CFG branches implement BranchOpInterface.
    Each region execution has independent registrations. This pass does not
    infer ownership; decomposition supplies region argument and result cleanup.
    Internal CFG forwarding remains borrowed.
    Re-entry into a registration block cleans up the previous iteration.
    Run after decomposition and before inlining or SCF-to-CFG lowering.
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect", "trident::tvm_ffi::abi::TVMFFIABIDialect"
  ];
}

def LowerTVMFFIABITensorMetadata : Pass<"lower-tvm-ffi-abi-tensor-metadata", "mlir::ModuleOp"> {
  let summary = "Lower TVM FFI tensor metadata operations through DLPack";
  let description = [{
    Rewrite TVMFFI tensor metadata operations into TVMFFI object views and
    DLPack tensor metadata operations.
  }];
  let dependentDialects = [
    "trident::dlpack::DLPackDialect", "trident::tvm_ffi::abi::TVMFFIABIDialect"
  ];
}

def LowerTVMFFIABIToPack : Pass<"lower-tvm-ffi-abi-to-pack", "mlir::ModuleOp"> {
  let summary = "Convert typed TVM FFI functions to packed entries";
  let description = [{
    Replace typed function bodies with packed entries, validating argument
    counts and tags before entering the original body. Requires decomposition.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect", "trident::tvm_ffi::abi::TVMFFIABIDialect"
  ];
}
