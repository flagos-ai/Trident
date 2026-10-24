//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def DecomposeTVMFFI : Pass<"decompose-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Lower semantic object types to their C ABI representations";
  let description = [{
    Preserve local calls while converting semantic containers to CArray,
    Tensor/Exception/Function to their Base types, and Str to CAny. Recursively convert signatures, regions and type-bearing attributes.
    Dedicated scf.if, scf.for and scf.while conversion patterns retain semantic
    exports and register argument/result releases in defer regions, including
    loop-carried values and zero-trip results. Every forwarded semantic slot
    retains a reference; producer cleanup remains local. Synchronous captures
    borrow outer references. Unsupported semantic region boundaries fail
    legalization; defer lowering validates cleanup region structure.
    Run after monad lowering and before defer lowering and inlining.
    Automatic lifetimes follow registration dominance scopes; Base IR remains
    manual.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def LowerTVMFFIDefer : Pass<"lower-tvm-ffi-defer", "mlir::ModuleOp"> {
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
    "mlir::cf::ControlFlowDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def LowerTVMFFIMonad : Pass<"lower-tvm-ffi-monad", "mlir::ModuleOp"> {
  let summary = "Lower Results to semantic Bool/Any tuples";
  let description = [{
    Lower monadic calls to separate Bool/Any results followed by explicit tuple
    construction, and expand monadic control flow, converting every Result to
    tuple<bool, any>. Run after kernel lowering and before container decomposition
    and cleanup. No monadic operations or Result types remain.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
    "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def LowerTensorMetadata : Pass<"lower-tensor-metadata", "mlir::ModuleOp"> {
  let summary = "Lower TVM FFI tensor metadata operations through DLPack";
  let description = [{
    Rewrite TVMFFI tensor metadata operations into TVMFFI object views and
    DLPack tensor metadata operations.
  }];
  let dependentDialects = [
    "trident::dlpack::DLPackDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def LowerToPack : Pass<"lower-to-pack", "mlir::ModuleOp"> {
  let summary = "Convert typed TVM FFI functions to packed entries";
  let description = [{
    Replace typed function bodies with packed entries, validating argument
    counts and tags before entering the original body. Requires decomposition.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}
