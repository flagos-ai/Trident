// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "trident/core/Dialect/TVMFFI/Monad/IR/TVMFFIOpBase.td"
def TVMFFIMonad_BindOp
    : TVMFFIMonad_Op<"bind", [SingleBlock, RecursiveMemoryEffects]> {
  let summary = "Run a continuation only when every input Result succeeds";
  // clang-format off
  let description = [{
    Input and output Result counts are independent and may be zero. With no
    inputs the body runs unconditionally. With no outputs the body is used only
    for effects, and failed inputs skip those effects without producing Results.
    Body arguments match input payloads and yielded Results match output types.
  }];
  // clang-format on
  let arguments = (ins Variadic<TVMFFIMonad_ResultType>:$inputs);
  let results = (outs Variadic<TVMFFIMonad_ResultType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "$inputs `:` functional-type($inputs, $outputs) $body attr-dict";
  // clang-format off
  let extraClassDeclaration = [{
    /// Return the entry block, creating it through the builder if needed.
    ::mlir::Block *addEntryBlock(::mlir::OpBuilder &builder);
    /// Return the entry block, or null when the body is empty.
    ::mlir::Block *getEntryBlock();
  }];
  // clang-format on
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def TVMFFIMonad_CallOp : TVMFFIMonad_Op<"call"> {
  let summary = "Call a local TVM FFI function and return its Result";
  let arguments = (ins FlatSymbolRefAttr:$callee,
      Variadic<AnyTVMFFIAnyABIType>:$operands,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs TVMFFIMonad_ResultType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` "
                       "functional-type($operands, $result)";
}

def TVMFFIMonad_FailOp : TVMFFIMonad_Op<"fail", [Pure]> {
  let summary = "Construct failed Results without changing the runtime error";
  let results = (outs Variadic<TVMFFIMonad_ResultType>:$outputs);
  let assemblyFormat = "attr-dict `:` type($outputs)";
  let hasVerifier = 1;
}

def TVMFFIMonad_PureOp
    : TVMFFIMonad_Op<
          "pure", [Pure,
                   TypesMatchWith<"payload type must match Result type",
                                  "output", "value",
                                  "::llvm::cast<::trident::tvm_ffi::monad::"
                                  "ResultType>($_self).getPayloadType()">]> {
  let summary = "Wrap one payload in a successful Result";
  // clang-format off
  let description = [{
    Borrows the payload. Lowering constructs an independently owned Result
    container whose runtime storage retains the payload.
  }];
  // clang-format on
  let arguments = (ins AnyType:$value);
  let results = (outs TVMFFIMonad_ResultType:$output);
  let assemblyFormat =
      "$value attr-dict `:` functional-type(operands, results)";
}

def TVMFFIMonad_UnpackOp
    : TVMFFIMonad_Op<
          "unpack", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary =
      "Expose Result success and its payload, widening tagged ABI values";
  // clang-format off
  let description = [{
    Returns success followed by the payload. Payloads supporting upcast to Any
    produce Any, with None on failure. Other payload types are preserved and
    their values may only be used on success. Does not clear runtime errors.
  }];
  // clang-format on
  let arguments = (ins TVMFFIMonad_ResultType:$input);
  let results = (outs I1:$success, AnyType:$value);
  let assemblyFormat =
      "$input attr-dict `:` functional-type(operands, results)";
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIMonad_YieldOp : TVMFFIMonad_Op<"yield", [Terminator]> {
  let summary = "Yield Results from a bind continuation";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

def TVMFFIMonad_DirectAllocationOp : TVMFFIMonad_Op<"direct_allocation"> {
  let arguments = (ins I32Attr:$kind, Variadic<AnyTVMFFIAnyABIType>:$arguments);
  let results = (outs TVMFFIMonad_ResultType:$result);
  let assemblyFormat = "$kind `(` $arguments `)` attr-dict `:` "
                       "functional-type($arguments, $result)";
}
