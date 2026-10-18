//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def DecomposeTVMFFI : Pass<"decompose-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Erase semantic containers and decompose TVM FFI binds";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect", "mlir::scf::SCFDialect",
    "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def OwnershipDeallocation : Pass<"ownership-deallocation", "mlir::ModuleOp"> {
  let summary = "Materialize TVM FFI object ownership in CFGs";
  let description = [{
    Analyze object ownership and liveness in function CFGs and insert the
    required ObjectIncRef and ObjectDecRef operations on exits and CFG edges.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}
