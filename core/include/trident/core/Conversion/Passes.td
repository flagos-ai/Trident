//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertDLPackToLLVM : Pass<"convert-dlpack-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower DLPack tensor metadata operations to LLVM dialect";
  let description =
      // clang-format off
      [{Lower DLPack tensor metadata operations to LLVM dialect.}];
  // clang-format on
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect",
                           "trident::dlpack::DLPackDialect"];
}

def ConvertTVMFFIABIToFunc
    : Pass<"convert-tvm-ffi-abi-to-func", "mlir::ModuleOp"> {
  let summary =
      "Convert local TVM FFI calls and packed functions to Func operations";
  let description =
      // clang-format off
      [{Convert packed TVM FFI entries and calls to status-returning Func operations, preserving success-only output access in SCF regions.}];
  // clang-format on
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "trident::tvm_ffi::abi::TVMFFIABIDialect"];
}

def ConvertTVMFFIABIToGPU
    : Pass<"convert-tvm-ffi-abi-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower TVM FFI kernel launches to GPU dialect ";
  // clang-format off
  let description = [{Lower TVM FFI kernel launches to GPU dialect.}];
  // clang-format on
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect", "mlir::gpu::GPUDialect",
                           "mlir::scf::SCFDialect",
                           "trident::dlpack::DLPackDialect",
                           "trident::tvm_ffi::abi::TVMFFIABIDialect"];
}

def ConvertTVMFFIABIToLLVM
    : Pass<"convert-tvm-ffi-abi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  // clang-format off
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
  // clang-format on
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::func::FuncDialect"];
}

def TridentLoweringPipeline
    : Pass<"trident-lowering-pipeline", "mlir::ModuleOp"> {
  let summary = "Lower TVM FFI and GPU operations to LLVM dialect";
  // clang-format off
  let description = [{Lower TVM FFI and GPU operations to LLVM dialect.}];
  // clang-format on
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def ConvertTVMFFIAPIToABI
    : Pass<"convert-tvm-ffi-api-to-abi", "mlir::ModuleOp"> {
  let summary = "Lower semantic object types to their C ABI representations";
  // clang-format off
  let description = [{
    Preserve local calls while converting semantic containers to CArray,
    Tensor/Exception/Function to their Base types, and Str to CAny. Recursively convert signatures, regions and type-bearing attributes.
    Dedicated scf.execute_region, scf.if, scf.for and scf.while patterns retain semantic
    exports and register argument/result releases in defer regions, including
    loop-carried values and zero-trip results. Every forwarded semantic slot
    retains a reference; producer cleanup remains local. Synchronous captures
    borrow outer references. Unsupported semantic region boundaries fail
    legalization; defer lowering validates cleanup region structure.
    Run after monad lowering and before defer lowering and inlining.
    Automatic lifetimes follow registration dominance scopes; Base IR remains
    manual.
  }];
  // clang-format on
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
                           "trident::tvm_ffi::abi::TVMFFIABIDialect",
                           "trident::tvm_ffi::api::TVMFFIAPIDialect",
                           "trident::tvm_ffi::monad::TVMFFIMonadDialect"];
}
