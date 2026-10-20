//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def DecomposeTVMFFI : Pass<"decompose-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Erase semantic container types";
  let description = [{
    Preserve local calls while converting semantic containers to CArray.
    Propagate explicit tvm_ffi.cleanup attributes from construction and extraction
    operations to their runtime calls without adding cleanup responsibility. Run before
    materialize-tvm-ffi-cleanup.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "trident::tvm_ffi::TVMFFIDialect"
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

def MaterializeTVMFFICleanup : Pass<"materialize-tvm-ffi-cleanup", "mlir::ModuleOp"> {
  let summary = "Retain exports and release owned references at block exits";
  let description = [{
    Run after container decomposition and before inlining. Consume
    tvm_ffi.retain on frontend monad.pure operations, retaining each
    borrowed payload occurrence before wrapping it. Low-level unmarked pure
    operations continue to transfer references.
    Consume tvm_ffi.retain on exits, retaining reference-bearing operands
    of func.return, tvm_ffi.return, scf.yield and tvm_ffi.monad.yield, including
    repeated operands.
    Consume tvm_ffi.cleanup markers, releasing each resource result once at its
    defining block exit in reverse operation and result order. Result payloads
    are unpacked and adjusted in independent scf.if success branches. Export
    retains precede all local releases.
    Process pure retains regardless of the parent region's block count. Process
    export retains and cleanup only when the immediate parent region has one
    block, without a scope operation whitelist. Preserve pending export and
    cleanup attributes directly in multi-block regions. Nested single-block
    regions may be processed independently; the walk driver skips unreachable blocks. Region-owning
    results are processed in the selected parent block. Assume input attributes
    are well-formed; run separate retain and cleanup pattern stages without
    folding. Remove consumed cleanup and export markers, including scalar-only
    and empty export markers, so repeated runs are idempotent. Unmarked exits
    request no export retains.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}
