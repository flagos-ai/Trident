// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

include "trident/core/Dialect/TVMFFI/ABI/IR/TVMFFIOpBase.td"

class TVMFFIABI_TensorIndexedMetadataOp<string mnemonic>
    : TVMFFIABI_Op<
          mnemonic, [MemoryEffects<[MemRead]>,
                     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins TVMFFIABI_TensorType:$tensor, I64:$index);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor)";
}

def TVMFFIABI_AsOp : TVMFFIABI_Op<"as", [Pure]> {
  let summary = "View the payload of a TVM FFI object as a native type";
  let arguments = (ins TVMFFIABI_ObjectType:$object);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$object attr-dict `:` type($object) `->` type($result)";
}

def TVMFFIABI_CallOp : TVMFFIABI_Op<"call"> {
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
      Variadic<AnyTVMFFIABIValueType>:$operands,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs TVMFFIABI_BoolType:$success, TVMFFIABI_AnyType:$value);
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

class TVMFFIABI_ConstantOp<string mnemonic, Type resultType, Attr valueType,
                           list<Trait> traits = []>
    : TVMFFIABI_Op<mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<
                                                     InferTypeOpInterface>])> {
  let arguments = (ins valueType:$value);
  let results = (outs resultType:$result);
  let assemblyFormat = "$value attr-dict";
}

def TVMFFIABI_ConstantBoolOp
    : TVMFFIABI_ConstantOp<"constant.bool", TVMFFIABI_BoolType,
                           I1Attr, [Pure]> {
  let summary = "Create a TVM FFI boolean constant";
}

def TVMFFIABI_ConstantDTypeOp
    : TVMFFIABI_ConstantOp<"constant.dtype", TVMFFIABI_DTypeType, ArrayAttr> {
  let summary = "Create a TVM FFI dtype constant";
  let hasVerifier = 1;
}

def TVMFFIABI_ConstantDeviceOp
    : TVMFFIABI_ConstantOp<"constant.device", TVMFFIABI_DeviceType, StrAttr> {
  let summary = "Create a TVM FFI device constant";
}

def TVMFFIABI_ConstantFloatOp
    : TVMFFIABI_ConstantOp<"constant.float", TVMFFIABI_FloatType,
                           F64Attr, [Pure]> {
  let summary = "Create a TVM FFI floating-point constant";
}

def TVMFFIABI_ConstantIntOp
    : TVMFFIABI_ConstantOp<"constant.int", TVMFFIABI_IntType, I64Attr, [Pure]> {
  let summary = "Create a TVM FFI integer constant";
}

def TVMFFIABI_ConstantNoneOp
    : TVMFFIABI_Op<
          "constant.none", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create a TVM FFI None constant";
  let results = (outs TVMFFIABI_NoneType:$result);
  let assemblyFormat = "attr-dict";
}

def TVMFFIABI_ConstantRawStrOp
    : TVMFFIABI_ConstantOp<"constant.raw_str", TVMFFIABI_RawStrType, StrAttr> {
  let summary = "Create a TVM FFI raw string constant";
}

def TVMFFIABI_DeferOp : TVMFFIABI_Op<"defer", [SingleBlock]> {
  let summary = "Execute cleanup when leaving the registration dominance scope";
  // clang-format off
  let description = [{
    Registers cleanup without executing it immediately. Cleanup regions capture
    external values and execute when control leaves the registering block's
    dominance scope. Relative execution order between defers at the same exit
    is unspecified; operations requiring an order must share one body. Body
    operation order is preserved. Re-entry into the registering block cleans up the previous
    iteration. The enclosing region must have SSA control flow and block
    terminators. Cleanup runs before any terminator without CFG successors,
    including transfers to another region. CFG branches use BranchOpInterface.
    Each execution of a region has independent registrations.
    CFG block arguments borrow; forwarding a value does not extend its lifetime.
    Lower before inlining or control-flow conversion. The body has
    no arguments and ends in defer_yield; nested regions are not supported.
    This operation deliberately retains conservative side effects.
  }];
  // clang-format on
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
  // clang-format off
  let extraClassDeclaration = [{
    /// Return the entry block, creating it through the builder if needed.
    ::mlir::Block *addEntryBlock(::mlir::OpBuilder &builder);
    /// Return the entry block, or null when the body is empty.
    ::mlir::Block *getEntryBlock();
  }];
  // clang-format on
  let hasVerifier = 1;
}

def TVMFFIABI_DeferYieldOp : TVMFFIABI_Op<"defer_yield", [Terminator]> {
  let summary = "End a deferred cleanup body";
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def TVMFFIABI_DowncastOp
    : TVMFFIABI_Op<"downcast", [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Check and narrow a TVM FFI value, or preserve its type";
  // clang-format off
  let description = [{
    Identical source and result types are a no-op. Otherwise the source must
    be Any or Union and the result must narrow it; a runtime tag mismatch traps.
  }];
  // clang-format on
  let arguments = (ins AnyTVMFFIABIValueType:$value);
  let results = (outs AnyTVMFFIABIValueType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def TVMFFIABI_EqOp
    : TVMFFIABI_Op<"eq", [Pure,
                          DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Compare the ABI representation of two TVM FFI values";
  let arguments = (ins AnyTVMFFIABIValueType:$lhs, AnyTVMFFIABIValueType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

def TVMFFIABI_ExceptionOp
    : TVMFFIABI_Op<
          "exception", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create a Trident FFI exception object";
  let arguments = (ins StrAttr:$kind);
  let results = (outs TVMFFIABI_ExceptionType:$result);
  let assemblyFormat = "$kind attr-dict `:` type($result)";
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIABI_FuncOp
    : TVMFFIABI_Op<"func", [AffineScope, AutomaticAllocationScope,
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
    tvm_ffi_abi.func @add(%a: !tvm_ffi_abi.int, %b: !tvm_ffi_abi.int) -> !tvm_ffi_abi.int {
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

def TVMFFIABI_FunctionCallOp : TVMFFIABI_Op<"FunctionCall"> {
  let summary = "Call a TVM FFI function handle";
  let arguments = (ins TVMFFIABI_FunctionType:$callee,
      Variadic<AnyTVMFFIABIValueType>:$arguments);
  let results = (outs AnyTVMFFIABIValueType:$result);
  let assemblyFormat = "$callee `:` type($callee) `(` $arguments `)` attr-dict "
                       "`:` functional-type($arguments, $result)";
}

def TVMFFIABI_FunctionGetGlobalOp
    : TVMFFIABI_Op<"FunctionGetGlobal", [DeclareOpInterfaceMethods<
                                            InferTypeOpInterface>]> {
  let summary = "Get a TVM FFI global function by name";
  let arguments = (ins StrAttr:$name);
  let results = (outs TVMFFIABI_FunctionType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIABI_GetOp
    : TVMFFIABI_Op<"get", [Pure,
                           DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extract a native value from a TVM FFI value";
  let arguments = (ins AnyTVMFFIABIGetOperandType:$operand);
  let results = (outs AnyTVMFFIABIGetResultType:$result);
  let hasFolder = 1;
  let hasVerifier = 1;
  let assemblyFormat =
      "$operand attr-dict `:` type($operand) `->` type($result)";
}

def TVMFFIABI_IsaOp : TVMFFIABI_Op<"isa", [Pure]> {
  let summary = "Test a tagged value's runtime type without trapping";
  // clang-format off
  let description = [{
    Tests a concrete runtime tag, not container element types. Targets must
    have one runtime type index, directly or after semantic decomposition.
  }];
  // clang-format on
  let arguments = (ins TVMFFIABI_AnyType:$value, TypeAttr:$target);
  let results = (outs I1:$result);
  let assemblyFormat = "$value `is` $target attr-dict `:` type($value)";
  let hasVerifier = 1;
}

// Define the operand constraint before the launch operation that uses it.

def TVMFFIABI_NativeKernelLaunchOp
    : TVMFFIABI_Op<
          "native_kernel_launch",
          [AttrSizedOperandSegments,
           AllTypesMatch<["gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX",
                          "blockSizeY", "blockSizeZ"]>]> {
  let summary = "Launch a compiled kernel with native ABI operands";
  let hasVerifier = 1;
  // clang-format off
  let description = [{
    Unconditionally launch with native ABI operands. Specialization guards and
    Result construction are supplied by lower-kernel-launch using Monad ops.
  }];
  // clang-format on

  let arguments = (ins SymbolRefAttr:$kernel, I64:$gridSizeX, I64:$gridSizeY,
      I64:$gridSizeZ, I64:$blockSizeX, I64:$blockSizeY, I64:$blockSizeZ,
      Optional<I64>:$clusterSizeX, Optional<I64>:$clusterSizeY,
      Optional<I64>:$clusterSizeZ, Optional<I32>:$dynamicSharedMemorySize,
      Variadic<AnyType>:$kernelOperands);
  // clang-format off
  let assemblyFormat = [{
    $kernel
    (`clusters` `in` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)`)?
    `blocks` `in` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
    `:` type($gridSizeX)
    `threads` `in` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
    `args` `(` $kernelOperands `:` type($kernelOperands) `)`
    prop-dict attr-dict
}];
  // clang-format on
}

def TVMFFIABI_ObjectDecRefOp : TVMFFIABI_Op<"ObjectDecRef"> {
  let summary = "Decrement a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIABIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFIABI_ObjectIncRefOp : TVMFFIABI_Op<"ObjectIncRef"> {
  let summary = "Increment a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIABIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFIABI_PackCallOp : TVMFFIABI_Op<"pack.call"> {
  let summary = "Forward a borrowed argument pack to a local packed ABI entry";
  let arguments = (ins FlatSymbolRefAttr:$callee,
      TVMFFIABI_PackArgumentsType:$arguments);
  let results = (outs TVMFFIABI_BoolType:$success, TVMFFIABI_AnyType:$value);
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

def TVMFFIABI_PackFuncOp
    : TVMFFIABI_Op<"pack.func", [AffineScope, AutomaticAllocationScope,
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

def TVMFFIABI_PackGetOp : TVMFFIABI_Op<"pack.get"> {
  let summary =
      "Borrow a tagged argument after asserting the index is in bounds";
  let arguments = (ins TVMFFIABI_PackArgumentsType:$arguments, I64:$index);
  let results = (outs TVMFFIABI_AnyType:$value);
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
  let assemblyFormat = "$arguments `[` $index `]` attr-dict `:` "
                       "functional-type(operands, results)";
}

def TVMFFIABI_PackSizeOp : TVMFFIABI_Op<"pack.size", [Pure]> {
  let summary = "Read the native i64 argument count";
  let arguments = (ins TVMFFIABI_PackArgumentsType:$arguments);
  let results = (outs I64:$size);
  let assemblyFormat = "$arguments attr-dict `:` type($arguments)";
}

def TVMFFIABI_ReturnOp
    : TVMFFIABI_Op<"return", [ParentOneOf<["FuncOp", "PackFuncOp"]>, ReturnLike,
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
  let arguments = (ins TVMFFIABI_BoolType:$success, TVMFFIABI_AnyType:$value);
  let assemblyFormat =
      "attr-dict $success `,` $value `:` type($success) `,` type($value)";
}

def TVMFFIABI_TensorCloneOp
    : TVMFFIABI_Op<
          "tensor.clone", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create an independent tensor copy preserving the layout";
  let arguments = (ins TVMFFIABI_TensorType:$tensor);
  let results = (outs TVMFFIABI_TensorType:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIABI_TensorCopyOp : TVMFFIABI_Op<"tensor.copy_", [SameTypeOperands]> {
  let summary = "Copy tensor contents into an existing tensor in place";
  let arguments = (ins TVMFFIABI_TensorType:$destination,
      TVMFFIABI_TensorType:$source);
  let assemblyFormat =
      "$destination `,` $source attr-dict `:` type($destination)";
}

def TVMFFIABI_TensorDTypeOp
    : TVMFFIABI_Op<
          "tensor.dtype", [MemoryEffects<[MemRead]>,
                           DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack dtype fields from a runtime tensor";
  let arguments = (ins TVMFFIABI_TensorType:$tensor);
  let results = (outs I8:$code, I8:$bits, I16:$lanes);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIABI_TensorDeviceOp
    : TVMFFIABI_Op<
          "tensor.device", [MemoryEffects<[MemRead]>,
                            DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack device fields from a runtime tensor";
  let arguments = (ins TVMFFIABI_TensorType:$tensor);
  let results = (outs I32:$device_type, I32:$device_index);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIABI_TensorDimOp
    : TVMFFIABI_Op<
          "tensor.dim", [MemoryEffects<[MemRead]>,
                         DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read the rank of a runtime tensor";
  let arguments = (ins TVMFFIABI_TensorType:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIABI_TensorLiteralOp
    : TVMFFIABI_Op<
          "tensor.literal", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create an owned TVM FFI tensor from a dense literal";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs TVMFFIABI_TensorType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
  // clang-format off
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  // clang-format on
}

def TVMFFIABI_TensorSizeOp : TVMFFIABI_TensorIndexedMetadataOp<"tensor.size"> {
  let summary = "Read a runtime tensor dimension size";
}

def TVMFFIABI_TensorStorageOffsetOp
    : TVMFFIABI_Op<"tensor.storage_offset", [MemoryEffects<[MemRead]>,
                                             DeclareOpInterfaceMethods<
                                                 InferTypeOpInterface>]> {
  let summary = "Read a runtime tensor storage offset";
  let arguments = (ins TVMFFIABI_TensorType:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFIABI_TensorStrideOp
    : TVMFFIABI_TensorIndexedMetadataOp<"tensor.stride"> {
  let summary = "Read a runtime tensor stride";
}

def TVMFFIABI_ToOp
    : TVMFFIABI_Op<"to", [Pure,
                          DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Convert a native scalar to a TVM FFI semantic value";
  let arguments = (ins AnyTypeOf<[F64, I1, I64]>:$value);
  let results = (outs TVMFFIABI_NumericType:$result);
  let hasFolder = 1;
  let hasVerifier = 1;
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
}

def TVMFFIABI_UpcastOp
    : TVMFFIABI_Op<"upcast", [Pure,
                              DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Widen a TVM FFI value, or preserve its type";
  // clang-format off
  let description = [{
    Identical source and result types are a no-op. Otherwise the result must
    be Any or a Union containing the source type.
  }];
  // clang-format on
  let arguments = (ins AnyTVMFFIABIValueType:$value);
  let results = (outs AnyTVMFFIABIValueType:$result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}
