//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def ConvertDLPackToLLVM : Pass<"convert-dlpack-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower DLPack tensor metadata operations to LLVM dialect";
  let description = [{Lower DLPack tensor metadata operations to LLVM dialect.}];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect",
    "trident::dlpack::DLPackDialect"
  ];
}

def ConvertTVMFFIToFunc : Pass<"convert-tvm-ffi-to-func", "mlir::ModuleOp"> {
  let summary = "Convert TVMFFI functions to Func operations";
  let description = [{Convert TVMFFI functions to Any ABI Func operations.}];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def ConvertTVMFFIToGPU : Pass<"convert-tvm-ffi-to-gpu", "mlir::ModuleOp"> {
  let summary = "Lower TVM FFI kernel launches to GPU dialect ";
  let description = [{Lower TVM FFI kernel launches to GPU dialect.}];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
    "mlir::gpu::GPUDialect", "trident::dlpack::DLPackDialect",
    "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def ConvertTVMFFIToLLVM : Pass<"convert-tvm-ffi-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower TVMFFI dialect operations to LLVM dialect";
  let description = [{Lower TVMFFI dialect operations to LLVM dialect.}];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect"
  ];
}

def FinalizeTVMFFI : Pass<"finalize-tvm-ffi", "mlir::ModuleOp"> {
  let summary = "Finalize TVMFFI tensor metadata operations through DLPack";
  let description = [{
    Rewrite TVMFFI tensor metadata operations into TVMFFI object views and
    DLPack tensor metadata operations.
  }];
  let dependentDialects = [
    "trident::dlpack::DLPackDialect", "trident::tvm_ffi::TVMFFIDialect"
  ];
}

def TridentLoweringPipeline
    : Pass<"trident-lowering-pipeline", "mlir::ModuleOp"> {
  let summary = "Lower TVM FFI and GPU operations to LLVM dialect";
  let description = [{Lower TVM FFI and GPU operations to LLVM dialect.}];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect", "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}
