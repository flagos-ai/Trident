cmake_minimum_required(VERSION 3.29)

project(TridentCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "try:
    from triton._flagtree_backend import FLAGTREE_BACKEND
except ImportError:
    FLAGTREE_BACKEND = ''
print(FLAGTREE_BACKEND)"
  OUTPUT_VARIABLE TRIDENT_FLAGTREE_BACKEND
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(TRIDENT_ASCEND_BACKEND OFF)
if(TRIDENT_FLAGTREE_BACKEND STREQUAL "ascend")
  set(TRIDENT_ASCEND_BACKEND ON)
else()
  find_package(CUDAToolkit REQUIRED)
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE TRIDENT_NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import contextlib, sys
with contextlib.redirect_stdout(sys.stderr):
    import torch
    prefix = torch.utils.cmake_prefix_path
print(prefix)"
  OUTPUT_VARIABLE TRIDENT_TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TRIDENT_TORCH_QUERY_RESULT
  ERROR_VARIABLE TRIDENT_TORCH_QUERY_ERROR)
if(NOT TRIDENT_TORCH_QUERY_RESULT STREQUAL "0")
  message(
    FATAL_ERROR
      "Failed to query PyTorch CMake prefix:\n${TRIDENT_TORCH_QUERY_ERROR}")
endif()
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import contextlib, sys
with contextlib.redirect_stdout(sys.stderr):
    import pathlib, tvm_ffi
    prefix = pathlib.Path(tvm_ffi.__file__).parent / 'share' / 'cmake'
print(prefix)"
  OUTPUT_VARIABLE TRIDENT_TVM_FFI_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TRIDENT_TVM_FFI_QUERY_RESULT
  ERROR_VARIABLE TRIDENT_TVM_FFI_QUERY_ERROR)
if(NOT TRIDENT_TVM_FFI_QUERY_RESULT STREQUAL "0")
  message(
    FATAL_ERROR
      "Failed to query tvm_ffi CMake prefix:\n${TRIDENT_TVM_FFI_QUERY_ERROR}")
endif()

find_package(MLIR REQUIRED CONFIG)
set(Caffe2_DIR
    "${TRIDENT_TORCH_CMAKE_PREFIX}/Caffe2"
    CACHE PATH "PyTorch Caffe2 CMake package directory" FORCE)
set(nanobind_DIR
    "${TRIDENT_NANOBIND_CMAKE_DIR}"
    CACHE PATH "nanobind CMake package directory" FORCE)
set(Torch_DIR
    "${TRIDENT_TORCH_CMAKE_PREFIX}/Torch"
    CACHE PATH "PyTorch CMake package directory" FORCE)
set(tvm_ffi_DIR
    "${TRIDENT_TVM_FFI_CMAKE_PREFIX}/tvm_ffi"
    CACHE PATH "tvm_ffi CMake package directory" FORCE)
find_package(nanobind CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(tvm_ffi CONFIG REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(HandleLLVMOptions)
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)

include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIRPython.cmake)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

include(CheckCXXSourceCompiles)
set(_trident_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_INCLUDES ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
check_cxx_source_compiles(
  [=[
#include <mlir/IR/OperationSupport.h>
int main() {
  mlir::PropertyRef *property = nullptr;
  return property != nullptr;
}
]=]
  TRIDENT_MLIR_HAS_PROPERTY_REF)
check_cxx_source_compiles(
  [=[
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <optional>
#include <utility>
int main() {
  using LaunchOp = decltype(mlir::gpu::LaunchFuncOp::create(
      std::declval<mlir::OpBuilder &>(), std::declval<mlir::Location>(),
      std::declval<mlir::SymbolRefAttr>(),
      std::declval<mlir::gpu::KernelDim3>(),
      std::declval<mlir::gpu::KernelDim3>(), std::declval<mlir::Value>(),
      std::declval<mlir::ValueRange>(), std::declval<mlir::Type>(),
      std::declval<mlir::ValueRange>(), std::declval<mlir::Value>(),
      std::declval<std::optional<mlir::gpu::KernelDim3>>()));
  return sizeof(LaunchOp) == 0;
}
]=]
  TRIDENT_MLIR_GPU_LAUNCH_HAS_ASYNC_DEPS_AND_OBJECT)
set(CMAKE_REQUIRED_INCLUDES "${_trident_saved_required_includes}")
unset(_trident_saved_required_includes)
if(TRIDENT_MLIR_HAS_PROPERTY_REF)
  add_compile_definitions(TRIDENT_MLIR_HAS_PROPERTY_REF)
endif()
if(TRIDENT_MLIR_GPU_LAUNCH_HAS_ASYNC_DEPS_AND_OBJECT)
  add_compile_definitions(TRIDENT_MLIR_GPU_LAUNCH_HAS_ASYNC_DEPS_AND_OBJECT)
endif()

if(NOT TRIDENT_ASCEND_BACKEND)
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
endif()
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories(SYSTEM ${tvm_ffi_INCLUDE_DIR})
include_directories(SYSTEM ${tvm_ffi_DLPACK_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)
