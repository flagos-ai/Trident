// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_MONAD_TVMFFIOPS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_MONAD_TVMFFIOPS_TD_

include "trident/core/Dialect/TVMFFI/IR/TVMFFIOpBase.td"

def TVMFFI_MonadBindOp : TVMFFI_Op<"monad.bind", [SingleBlock, RecursiveMemoryEffects]> {
  let summary = "Run a continuation only when every input Result succeeds";
  let description = [{
    Input and output Result counts are independent and may be zero. With no
    inputs the body runs unconditionally. With no outputs the body is used only
    for effects, and failed inputs skip those effects without producing Results.
    Body arguments match input payloads and yielded Results match output types.
  }];
  let arguments = (ins Variadic<TVMFFI_ResultType>:$inputs);
  let results = (outs Variadic<TVMFFI_ResultType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$inputs `:` functional-type($inputs, $outputs) $body attr-dict";
  let extraClassDeclaration = [{
    /// Return the entry block, creating it through the builder if needed.
    ::mlir::Block *addEntryBlock(::mlir::OpBuilder &builder);
    /// Return the entry block, or null when the body is empty.
    ::mlir::Block *getEntryBlock();
  }];
  let hasVerifier = 1;
}

def TVMFFI_MonadCallOp : TVMFFI_Op<"monad.call"> {
  let summary = "Call a local TVM FFI function and return its Result";
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyTVMFFIAnyABIType>:$operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs TVMFFI_ResultType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` "
                       "functional-type($operands, $result)";
}

def TVMFFI_MonadFailOp : TVMFFI_Op<"monad.fail", [Pure]> {
  let summary = "Construct failed Results without changing the runtime error";
  let results = (outs Variadic<TVMFFI_ResultType>:$outputs);
  let assemblyFormat = "attr-dict `:` type($outputs)";
  let hasVerifier = 1;
}

def TVMFFI_MonadPureOp : TVMFFI_Op<"monad.pure", [Pure]> {
  let summary = "Wrap payloads in successful Results";
  let description = [{
    Borrows each payload occurrence. Lowering constructs independently owned
    Result containers whose runtime storage retains the payloads.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs Variadic<TVMFFI_ResultType>:$outputs);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $outputs)";
  let hasVerifier = 1;
}

def TVMFFI_MonadUnpackOp : TVMFFI_Op<"monad.unpack", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Expose Result success and its payload, widening tagged ABI values";
  let description = [{
    Returns success followed by the payload. Payloads supporting upcast to Any
    produce Any, with None on failure. Other payload types are preserved and
    their values may only be used on success. Does not clear runtime errors.
  }];
  let arguments = (ins TVMFFI_ResultType:$input);
  let results = (outs I1:$success, AnyType:$value);
  let assemblyFormat = "$input attr-dict `:` functional-type(operands, results)";
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
}

def TVMFFI_MonadYieldOp
    : TVMFFI_Op<"monad.yield", [Terminator]> {
  let summary = "Yield Results from a bind continuation";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

#endif
