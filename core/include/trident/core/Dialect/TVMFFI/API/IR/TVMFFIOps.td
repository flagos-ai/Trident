// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "trident/core/Dialect/TVMFFI/API/IR/TVMFFIOpBase.td"

class TVMFFIAPI_TensorIndexedMetadataOp<string mnemonic>
    : TVMFFIAPI_Op<
          mnemonic, [MemoryEffects<[MemRead]>,
                     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins TVMFFIAPI_TensorValue:$tensor, I64:$index);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor)";
}

def TVMFFIAPI_ArrayCreateOp : TVMFFIAPI_Op<"array.create"> {
  let arguments = (ins Variadic<AnyTVMFFISemanticType>:$elements);
  let results = (outs TVMFFIAPI_ArrayType:$result);
  let hasVerifier = 1;
  let assemblyFormat =
      "`(` $elements `)` attr-dict `:` functional-type($elements, $result)";
}

def TVMFFIAPI_ArrayGetOp
    : TVMFFIAPI_Op<"array.get", [TypesMatchWith<
                                    "result must match array element type",
                                    "container", "result",
                                    "::llvm::cast<::trident::tvm_ffi::api::"
                                    "ArrayType>($_self).getElementType()">]> {
  let arguments = (ins TVMFFIAPI_ArrayType:$container, I64:$index);
  let assemblyFormat = "$container `[` $index `]` attr-dict `:` "
                       "type($container) `->` type($result)";
  let results = (outs AnyTVMFFISemanticType:$result);
}

def TVMFFIAPI_ArraySizeOp : TVMFFIAPI_Op<"array.size"> {
  let arguments = (ins TVMFFIAPI_ArrayType:$container);
  let results = (outs TVMFFIABI_IntType:$result);
  let assemblyFormat =
      "$container attr-dict `:` type($container) `->` type($result)";
}

def TVMFFIAPI_CallOp : TVMFFIAPI_Op<"call"> {
  let summary = "Call a local TVM FFI function and return success and payload";
  // clang-format off
  let description = [{
    Returns separate Bool and Any values: (true, payload) on success or
    (false, None) on failure. Failure preserves the runtime error. The payload
    owns the returned reference; the success flag needs no reference management.
    LLVM conversion expands the packed ABI without constructing a container.
  }];
  // clang-format on
  let arguments = (ins FlatSymbolRefAttr:$callee,
      Variadic<AnyTVMFFIAnyABIType>:$operands,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs TVMFFIABI_BoolType:$success, TVMFFIAPI_AnyValue:$value);
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
  let hasVerifier = 1;
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` "
                       "functional-type($operands, results)";
}

class TVMFFIAPI_ConstantOp<string mnemonic, Type resultType, Attr valueType,
                           list<Trait> traits = []>
    : TVMFFIAPI_Op<mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<
                                                     InferTypeOpInterface>])> {
  let arguments = (ins valueType:$value);
  let results = (outs resultType:$result);
  let assemblyFormat = "$value attr-dict";
}

def TVMFFIAPI_ConstantStrOp
    : TVMFFIAPI_ConstantOp<"constant.str", TVMFFIAPI_StrType, StrAttr> {
  let summary = "Create a semantic string constant";
}

// Deliberately not Pure: a failed check must trap even if the result is unused.

def TVMFFIAPI_DowncastOp
    : TVMFFIAPI_Op<"downcast", [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Check and narrow a TVM FFI value, or preserve its type";
  // clang-format off
  let description = [{
    Identical source and result types are a no-op. Otherwise the source must
    be Any or Union and the result must narrow it; a runtime tag mismatch traps.
  }];
  // clang-format on
  let arguments = (ins AnyTVMFFIAnyABIType:$value);
  let results = (outs AnyTVMFFIAnyABIType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def TVMFFIAPI_EqOp
    : TVMFFIAPI_Op<"eq", [Pure,
                          DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Compare the ABI representation of two TVM FFI values";
  let arguments = (ins AnyTVMFFIAnyABIType:$lhs, AnyTVMFFIAnyABIType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

def TVMFFIAPI_ExceptionOp
    : TVMFFIAPI_Op<
          "exception", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create a Trident FFI exception object";
  let arguments = (ins StrAttr:$kind);
  let results = (outs TVMFFIAPI_ExceptionValue:$result);
  let assemblyFormat = "$kind attr-dict `:` type($result)";
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIAPI_FuncOp
    : TVMFFIAPI_Op<"func", [AffineScope, AutomaticAllocationScope,
                            FunctionOpInterface, IsolatedFromAbove,
                            OpAsmOpInterface]> {
  let summary = "A TVM FFI function with one C ABI return";
  // clang-format off
  let description = [{
    Defines a function that can be called through the TVM FFI interface.
    The function must return exactly one C ABI value. Pack multiple values
    into a tuple payload and use a None payload for functions with no value.

    Example:

    ```mlir
    tvm_ffi_api.func @add(%a: !tvm_ffi_abi.int, %b: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
      ...
    }
    ```
  }];
  // clang-format on

  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins "::llvm::StringRef":$name,
      "::mlir::FunctionType":$type,
      CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>];
  // clang-format off
  let extraClassDeclaration = [{
    // Return the existing entry block, or create one from the function inputs.
    ::mlir::Block *addEntryBlock(::mlir::OpBuilder &builder);
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
    ::mlir::Region *getCallableRegion();
    // Return nullptr for an external function.
    ::mlir::Block *getEntryBlock();
    ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  }];
  // clang-format on

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def TVMFFIAPI_FunctionCallOp : TVMFFIAPI_Op<"function.call"> {
  let summary = "Call a TVM FFI function handle";
  let arguments = (ins TVMFFIAPI_FunctionValue:$callee,
      Variadic<AnyTVMFFISemanticType>:$arguments);
  let results = (outs AnyTVMFFISemanticType:$result);
  let assemblyFormat = "$callee `:` type($callee) `(` $arguments `)` attr-dict "
                       "`:` functional-type($arguments, $result)";
}

def TVMFFIAPI_FunctionGetGlobalOp
    : TVMFFIAPI_Op<"function.get_global", [DeclareOpInterfaceMethods<
                                              InferTypeOpInterface>]> {
  let summary = "Get a TVM FFI global function by name";
  let arguments = (ins StrAttr:$name);
  let results = (outs TVMFFIAPI_FunctionValue:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIAPI_GetOp
    : TVMFFIAPI_Op<"get", [Pure,
                           DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extract a native value from a TVM FFI value";
  let arguments = (ins AnyTVMFFIGetOperandType:$operand);
  let results = (outs AnyTVMFFIGetResultType:$result);
  let hasFolder = 1;
  let hasVerifier = 1;
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
}

def TVMFFIAPI_GlobalCallOp : TVMFFIAPI_Op<"global.call"> {
  let summary = "Call a TVM FFI function by its registry name";
  // clang-format off
  let description = [{
    Calls the function registered under the constant string name and returns
    one payload of the declared type. Parameters and results follow the same
    semantic type constraints as function.call. No-data results use None;
    multiple values follow the callee's container return convention.

    Lookup status, a non-null handle and call status are asserted before using
    the result. This operation does not return a Result or resolve a local
    symbol. It may have arbitrary callee side effects, even when unused.

    API-to-ABI conversion expands the lookup and call, releases the temporary
    handle and manages semantic result ownership. Global handle caching follows
    the existing ABI lookup policy. The declared signature is not inferred or
    validated against the registry.
  }];
  // clang-format on
  let arguments = (ins StrAttr:$name,
      Variadic<AnyTVMFFISemanticType>:$arguments);
  let results = (outs AnyTVMFFISemanticType:$result);
  let assemblyFormat = "$name `(` $arguments `)` attr-dict `:` "
                       "functional-type($arguments, $result)";
}

def TVMFFIAPI_IsaOp : TVMFFIAPI_Op<"isa", [Pure]> {
  let summary = "Test a tagged value's runtime type without trapping";
  // clang-format off
  let description = [{
    Tests a concrete runtime tag, not container element types. Targets must
    have one runtime type index, directly or after semantic decomposition.
  }];
  // clang-format on
  let arguments = (ins TVMFFIAPI_AnyValue:$value, TypeAttr:$target);
  let results = (outs I1:$result);
  let assemblyFormat = "$value `is` $target attr-dict `:` type($value)";
  let hasVerifier = 1;
}

// Define the operand constraint before the launch operation that uses it.

def TVMFFIAPI_KernelOperandType
    : AnyTypeOf<[AnyTVMFFISemanticType, F64, I1, I64],
                "TVM FFI value or native scalar accepted by a Triton launch">;

def TVMFFIAPI_KernelLaunchOp
    : TVMFFIAPI_Op<
          "kernel_launch", [AttrSizedOperandSegments,
                            AllTypesMatch<["gridSizeX", "gridSizeY",
                                           "gridSizeZ", "blockSizeX",
                                           "blockSizeY", "blockSizeZ"]>]> {
  let summary = "Launch a compiled kernel entry point";
  // clang-format off
  let description = [{
    Returns Result<None>: success after launching, failure on specialization
    rejection without launching. Failure does not set a runtime error. The
    caller propagates rejection through Results and handles it at its boundary.
    Success does not imply asynchronous GPU execution has completed.

    Models launching a compiled kernel symbol with explicit grid/block sizes.
    Operands use TVM FFI semantic values or native scalars. Each operand's
    specialization attribute defines its runtime precondition. Constant
    specializations are guard-only, while variable specializations are passed
    to the compiled Triton kernel.

    Optionally supports cluster-level parallelism via `clusterSizeX`,
    `clusterSizeY`, and `clusterSizeZ` arguments. When present, the kernel
    is launched with the specified thread-block cluster size. This feature
    is exclusive to certain architectures (e.g. NVIDIA Hopper and later).
}];
  // clang-format on

  let arguments = (ins SymbolRefAttr:$kernel, I64:$gridSizeX, I64:$gridSizeY,
      I64:$gridSizeZ, I64:$blockSizeX, I64:$blockSizeY, I64:$blockSizeZ,
      Optional<I64>:$clusterSizeX, Optional<I64>:$clusterSizeY,
      Optional<I64>:$clusterSizeZ, Optional<I32>:$dynamicSharedMemorySize,
      Variadic<TVMFFIAPI_KernelOperandType>:$kernelOperands,
      TVMFFIAPI_KernelSpecializationArrayAttr:$specializations);
  let results = (outs TVMFFI_ResultType:$result);
  let hasVerifier = 1;
  // clang-format off
let extraClassDeclaration = [{
    /// Returns true if cluster size is specified.
    bool hasClusterSize();
    static ::mlir::ParseResult parseKernelArguments(
        ::mlir::OpAsmParser &parser,
        ::llvm::SmallVectorImpl<::mlir::OpAsmParser::UnresolvedOperand> &operands,
        ::llvm::SmallVectorImpl<::mlir::Type> &types,
        ::mlir::ArrayAttr &specializations);
    static void printKernelArguments(
        ::mlir::OpAsmPrinter &printer,
        ::mlir::Operation *op [[maybe_unused]],
        ::mlir::OperandRange operands, ::mlir::TypeRange types,
        ::mlir::ArrayAttr specializations);
  }];
  // clang-format on
  // clang-format off
  let assemblyFormat = [{
    $kernel
    (`clusters` `in` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)`)?
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `:` type($gridSizeX)
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    `args` `(` custom<KernelArguments>($kernelOperands, type($kernelOperands), $specializations) `)`
    `->` type($result)
    prop-dict attr-dict
}];
  // clang-format on
}

def TVMFFIAPI_PackCallOp : TVMFFIAPI_Op<"pack.call"> {
  let summary = "Forward a borrowed argument pack to a local packed ABI entry";
  let arguments = (ins FlatSymbolRefAttr:$callee,
      TVMFFIABI_PackArgumentsType:$arguments);
  let results = (outs TVMFFIABI_BoolType:$success, TVMFFIAPI_AnyValue:$value);
  let hasVerifier = 1;
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
  let assemblyFormat = "$callee `(` $arguments `)` attr-dict `:` "
                       "functional-type(operands, results)";
}

def TVMFFIAPI_PackFuncOp
    : TVMFFIAPI_Op<"pack.func", [AffineScope, AutomaticAllocationScope,
                                 FunctionOpInterface, IsolatedFromAbove,
                                 OpAsmOpInterface]> {
  let summary = "A borrowed packed entry with one C ABI return";
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins "::llvm::StringRef":$name,
      "::mlir::FunctionType":$type,
      CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>];
  // clang-format off
  let extraClassDeclaration = [{
    // Return the existing entry block, or create one from the function inputs.
    ::mlir::Block *addEntryBlock(::mlir::OpBuilder &builder);
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
    ::mlir::Region *getCallableRegion();
    // Return nullptr for an external function.
    ::mlir::Block *getEntryBlock();
    ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  }];
  // clang-format on

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def TVMFFIAPI_PackGetOp : TVMFFIAPI_Op<"pack.get"> {
  let summary =
      "Borrow a tagged argument after asserting the index is in bounds";
  let arguments = (ins TVMFFIABI_PackArgumentsType:$arguments, I64:$index);
  let results = (outs TVMFFIAPI_AnyValue:$value);
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
  let assemblyFormat = "$arguments `[` $index `]` attr-dict `:` "
                       "functional-type(operands, results)";
}

def TVMFFIAPI_PackSizeOp : TVMFFIAPI_Op<"pack.size", [Pure]> {
  let summary = "Read the native i64 argument count";
  let arguments = (ins TVMFFIABI_PackArgumentsType:$arguments);
  let results = (outs I64:$size);
  let assemblyFormat = "$arguments attr-dict `:` type($arguments)";
}

def TVMFFIAPI_ReturnOp
    : TVMFFIAPI_Op<"return", [ParentOneOf<["FuncOp", "PackFuncOp"]>, ReturnLike,
                              Terminator]> {
  let summary =
      "Return explicit success and an Any payload from a C ABI function";
  // clang-format off
  let description = [{
    The function signature declares the successful C ABI payload type. Return
    carries a boxed success flag and an Any payload of that type on success,
    or None on failure. ABI lowering stores the payload only on success and
    preserves the runtime error on failure.
  }];
  // clang-format on
  let arguments = (ins TVMFFIABI_BoolType:$success, TVMFFIAPI_AnyValue:$value);
  let assemblyFormat =
      "attr-dict $success `,` $value `:` type($success) `,` type($value)";
}

def TVMFFIAPI_TensorCloneOp
    : TVMFFIAPI_Op<
          "tensor.clone", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create an independent tensor copy preserving the layout";
  let arguments = (ins TVMFFIAPI_TensorValue:$tensor);
  let results = (outs TVMFFIAPI_TensorValue:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIAPI_TensorCopyOp : TVMFFIAPI_Op<"tensor.copy_", [SameTypeOperands]> {
  let summary = "Copy tensor contents into an existing tensor in place";
  let arguments = (ins TVMFFIAPI_TensorValue:$destination,
      TVMFFIAPI_TensorValue:$source);
  let assemblyFormat =
      "$destination `,` $source attr-dict `:` type($destination)";
}

def TVMFFIAPI_TensorDTypeOp
    : TVMFFIAPI_Op<
          "tensor.dtype", [MemoryEffects<[MemRead]>,
                           DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack dtype fields from a runtime tensor";
  let arguments = (ins TVMFFIAPI_TensorValue:$tensor);
  let results = (outs I8:$code, I8:$bits, I16:$lanes);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIAPI_TensorDeviceOp
    : TVMFFIAPI_Op<
          "tensor.device", [MemoryEffects<[MemRead]>,
                            DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack device fields from a runtime tensor";
  let arguments = (ins TVMFFIAPI_TensorValue:$tensor);
  let results = (outs I32:$device_type, I32:$device_index);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIAPI_TensorDimOp
    : TVMFFIAPI_Op<
          "tensor.dim", [MemoryEffects<[MemRead]>,
                         DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read the rank of a runtime tensor";
  let arguments = (ins TVMFFIAPI_TensorValue:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIAPI_TensorLiteralOp
    : TVMFFIAPI_Op<
          "tensor.literal", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create an owned TVM FFI tensor from a dense literal";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs TVMFFIAPI_TensorValue:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIAPI_TensorSizeOp : TVMFFIAPI_TensorIndexedMetadataOp<"tensor.size"> {
  let summary = "Read a runtime tensor dimension size";
}

def TVMFFIAPI_TensorStorageOffsetOp
    : TVMFFIAPI_Op<"tensor.storage_offset", [MemoryEffects<[MemRead]>,
                                             DeclareOpInterfaceMethods<
                                                 InferTypeOpInterface>]> {
  let summary = "Read a runtime tensor storage offset";
  let arguments = (ins TVMFFIAPI_TensorValue:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIAPI_TensorStrideOp
    : TVMFFIAPI_TensorIndexedMetadataOp<"tensor.stride"> {
  let summary = "Read a runtime tensor stride";
}

def TVMFFIAPI_TupleConstructOp
    : TVMFFIAPI_Op<
          "tuple.construct", [Pure, RangedTypesMatchWith<
                                        "elements must match the tuple types",
                                        "result", "elements",
                                        "::llvm::cast<::trident::tvm_ffi::api::"
                                        "TupleType>($_self).getTypes()">]> {
  let summary = "Pack elements into a semantic tuple";
  let arguments = (ins Variadic<AnyTVMFFISemanticType>:$elements);
  let results = (outs TVMFFIAPI_TupleType:$result);
  let assemblyFormat =
      "`(` $elements `)` attr-dict `:` functional-type($elements, $result)";
}

def TVMFFIAPI_TupleUnconstructOp
    : TVMFFIAPI_Op<"tuple.unconstruct",
                   [Pure,
                    RangedTypesMatchWith<"results must match the tuple types",
                                         "container", "elements",
                                         "::llvm::cast<::trident::tvm_ffi::api:"
                                         ":TupleType>($_self).getTypes()">]> {
  let summary = "Unpack all semantic tuple elements in order";
  let hasFolder = 1;
  let arguments = (ins TVMFFIAPI_TupleType:$container);
  let results = (outs Variadic<AnyTVMFFISemanticType>:$elements);
  let assemblyFormat =
      "$container attr-dict `:` functional-type($container, $elements)";
}

def TVMFFIAPI_UpcastOp
    : TVMFFIAPI_Op<"upcast", [Pure,
                              DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Widen a TVM FFI value, or preserve its type";
  // clang-format off
  let description = [{
    Identical source and result types are a no-op. Otherwise the result must
    be Any or a Union containing the source type.
  }];
  // clang-format on
  let arguments = (ins AnyTVMFFIAnyABIType:$value);
  let results = (outs AnyTVMFFIAnyABIType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}
// A status-returning direct allocation. rank=-1 borrows shape arrays;
// nonnegative rank passes flattened native dimensions to the C ABI.
def TVMFFIAPI_DirectAllocationOp : TVMFFIAPI_Op<"direct_allocation"> {
  let arguments = (ins I32Attr:$kind, I64Attr:$rank,
      Variadic<AnyTVMFFISemanticType>:$arguments);
  let results = (outs TVMFFIABI_BoolType:$success, TVMFFIAPI_AnyValue:$value);
  let assemblyFormat = "$kind `rank` $rank `(` $arguments `)` attr-dict `:` "
                       "functional-type($arguments, results)";
}
