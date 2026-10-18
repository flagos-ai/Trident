//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <tuple>

namespace trident::conversion::utils {

/// Emit an assertion in every build mode and resume at the success block.
mlir::LogicalResult emitLLVMAssert(mlir::RewriterBase &rewriter,
                                   mlir::Location loc, mlir::Value condition);

/// Allocates a null-terminated string on the stack and returns a pointer to
/// its first byte.
mlir::Value getString(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef content);

/// Get an owned TVM FFI global function handle by name.
///
/// Resolved handles are cached at module initialization and released at unload.
/// Call sites retain a cached handle for the caller. A null cache entry sets a
/// RuntimeError on the calling thread without retaining the null handle.
///
/// \return The owned function handle and i32 status (zero on success).
mlir::FailureOr<std::tuple<mlir::Value, mlir::Value>>
getTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::ModuleOp moduleOp, llvm::StringRef funcName);

/// Call a borrowed TVM FFI function handle.
mlir::FailureOr<mlir::Value>
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   llvm::ArrayRef<mlir::Value> args, mlir::Value resultSlot);

mlir::FailureOr<mlir::Value>
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   mlir::Value argsArray, mlir::Value numArgs,
                   mlir::Value resultSlot);

/// Call a TVM FFI global function by name.
///
/// \param builder   The op builder (insertion point must be valid).
/// \param loc       Source location for generated ops.
/// \param moduleOp  Parent module (for declaring LLVM function symbols).
/// \param funcName  The TVM FFI function name (e.g. "ffi.Array").
/// \param args      Pre-built TVMFFIAny* slots (each a !llvm.ptr to an
///                  alloca'd {i32, i32, i64}).
/// \return A pointer to the result TVMFFIAny slot (!llvm.ptr to {i32,i32,i64})
///         on the stack, or failure.  The caller extracts the desired field
///         (e.g. field[2] for v_int64/v_obj).
mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         llvm::ArrayRef<mlir::Value> args);

/// Call a TVM FFI global function with a pre-built contiguous args array
/// and a runtime-determined number of arguments.
///
/// The caller provides the args array already populated instead of individual
/// slots, and \p numArgs is a runtime i32 Value instead of a compile-time
/// constant. The function handle comes from the module-level cache.
///
/// \param argsArray A !llvm.ptr to a contiguous array of TVMFFIAny elements.
/// \param numArgs   A runtime i32 value specifying how many elements to pass.
/// \return A pointer to the result TVMFFIAny slot (!llvm.ptr to {i32,i32,i64})
///         on the stack, or failure.
mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         mlir::Value argsArray, mlir::Value numArgs);

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
