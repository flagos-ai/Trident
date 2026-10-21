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
    Tensor/Exception/Function to their Base types, and Str to a representation
    Union. Recursively convert signatures, regions and type-bearing attributes.
    Automatically retain semantic exports and register local releases in defer
    regions. Run after monad lowering and before defer lowering and inlining.
    Automatic lifetimes require single-block regions; Base IR remains manual.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def LowerTVMFFIDefer : Pass<"lower-tvm-ffi-defer", "mlir::ModuleOp"> {
  let summary = "Expand deferred cleanup before the current block terminator";
  let description = [{
    Move defer bodies before their enclosing block terminator in reverse
    registration order, preserving operation order within each body and running
    after existing export retains. Require single-block tvm_ffi.func, func.func
    or scf.if regions with tvm_ffi.return, func.return or scf.yield exits.
    Run after decomposition and before inlining or SCF-to-CFG lowering.
  }];
  let dependentDialects = ["trident::tvm_ffi::TVMFFIDialect"];
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
