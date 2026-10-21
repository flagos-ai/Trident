//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_

include "trident/core/Dialect/TVMFFI/IR/TVMFFIOpBase.td"

class TVMFFI_TensorIndexedMetadataOp<string mnemonic>
    : TVMFFI_Op<mnemonic, [Pure,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins TVMFFI_TensorValue : $tensor, I64 : $index);
  let results = (outs I64 : $result);
  let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor)";
}

def TVMFFI_ArrayCreateOp : TVMFFI_Op<"array.create"> {
  let arguments = (ins Variadic<AnyTVMFFISemanticType>:$elements);
  let results = (outs TVMFFI_ArrayType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "`(` $elements `)` attr-dict `:` functional-type($elements, $result)";
}

def TVMFFI_ArrayGetOp : TVMFFI_Op<"array.get", [
    TypesMatchWith<"result must match array element type", "container", "result",
      "::llvm::cast<::trident::tvm_ffi::ArrayType>($_self).getElementType()">
]> {
  let arguments = (ins TVMFFI_ArrayType:$container, I64:$index);
  let assemblyFormat = "$container `[` $index `]` attr-dict `:` type($container) `->` type($result)";
  let results = (outs AnyTVMFFISemanticType:$result);
}

def TVMFFI_ArraySizeOp : TVMFFI_Op<"array.size"> {
  let arguments = (ins TVMFFI_ArrayType:$container);
  let results = (outs TVMFFI_IntType:$result);
  let assemblyFormat = "$container attr-dict `:` type($container) `->` type($result)";
}

def TVMFFI_AsOp : TVMFFI_Op<"as", [Pure]> {
  let summary = "View the payload of a TVM FFI object as a native type";
  let arguments = (ins TVMFFI_ObjectType:$object);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$object attr-dict `:` type($object) `->` type($result)";
}

def TVMFFI_CallOp : TVMFFI_Op<"call"> {
  let summary = "Call a local TVM FFI function and return success and payload";
  let description = [{
    Returns separate Bool and Any values: (true, payload) on success or
    (false, None) on failure. Failure preserves the runtime error. The payload
    owns the returned reference; the success flag needs no reference management.
    LLVM conversion expands the packed ABI without constructing a container.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyTVMFFIAnyABIType>:$operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs TVMFFI_BoolType:$success, TVMFFI_AnyValue:$value);
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
  let hasVerifier = 1;
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` "
                       "functional-type($operands, results)";
}

class TVMFFI_ConstantOp<string mnemonic, Type resultType, Attr valueType>
    : TVMFFI_Op<mnemonic, [
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins valueType:$value);
  let results = (outs resultType:$result);
  let assemblyFormat = "$value attr-dict";
}

def TVMFFI_ConstantBoolOp
    : TVMFFI_ConstantOp<"constant.bool", TVMFFI_BoolType, I1Attr> {
  let summary = "Create a TVM FFI boolean constant";
}

def TVMFFI_ConstantDTypeOp
    : TVMFFI_ConstantOp<"constant.dtype", TVMFFI_DTypeType, ArrayAttr> {
  let summary = "Create a TVM FFI dtype constant";
  let hasVerifier = 1;
}

def TVMFFI_ConstantDeviceOp
    : TVMFFI_ConstantOp<"constant.device", TVMFFI_DeviceType, StrAttr> {
  let summary = "Create a TVM FFI device constant";
}

def TVMFFI_ConstantFloatOp
    : TVMFFI_ConstantOp<"constant.float", TVMFFI_FloatType, F64Attr> {
  let summary = "Create a TVM FFI floating-point constant";
}

def TVMFFI_ConstantIntOp
    : TVMFFI_ConstantOp<"constant.int", TVMFFI_IntType, I64Attr> {
  let summary = "Create a TVM FFI integer constant";
}

def TVMFFI_ConstantNoneOp
    : TVMFFI_Op<"constant.none", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create a TVM FFI None constant";
  let results = (outs TVMFFI_NoneType:$result);
  let assemblyFormat = "attr-dict";
}

def TVMFFI_ConstantRawStrOp
    : TVMFFI_ConstantOp<"constant.raw_str", TVMFFI_RawStrType, StrAttr> {
  let summary = "Create a TVM FFI raw string constant";
}

def TVMFFI_ConstantStrOp
    : TVMFFI_ConstantOp<"constant.str", TVMFFI_StrType, StrAttr> {
  let summary = "Create a semantic string constant";
}

// Deliberately not Pure: a failed check must trap even if the result is unused.
def TVMFFI_DeferOp : TVMFFI_Op<"defer", [SingleBlock]> {
  let summary = "Execute cleanup before leaving the current block";
  let description = [{
    Registers cleanup without executing it immediately. Cleanup regions capture
    external values and execute before the current block terminator in reverse
    registration order, preserving operation order within each body. The enclosing
    region must be single-block and belong to tvm_ffi.func, func.func or scf.if,
    exiting through tvm_ffi.return, func.return or scf.yield.
    Lower before inlining or control-flow conversion. The body has
    no arguments and ends in defer_yield; nested regions are not supported.
    This operation deliberately retains conservative side effects.
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
  let hasVerifier = 1;
}

def TVMFFI_DeferYieldOp : TVMFFI_Op<"defer_yield", [Terminator]> {
  let summary = "End a deferred cleanup body";
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def TVMFFI_DowncastOp
    : TVMFFI_Op<"downcast", [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Check and narrow a TVM FFI value, or preserve its type";
  let description = [{
    Identical source and result types are a no-op. Otherwise the source must
    be Any or Union and the result must narrow it; a runtime tag mismatch traps.
  }];
  let arguments = (ins AnyTVMFFIAnyABIType : $value);
  let results = (outs AnyTVMFFIAnyABIType : $result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}

def TVMFFI_EqOp : TVMFFI_Op<"eq", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Compare the ABI representation of two TVM FFI values";
  let arguments = (ins AnyTVMFFIAnyABIType : $lhs,
                       AnyTVMFFIAnyABIType : $rhs);
  let results = (outs I1 : $result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

def TVMFFI_ExceptionOp
    : TVMFFI_Op<"exception", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create a Trident FFI exception object";
  let arguments = (ins StrAttr : $kind);
  let results = (outs TVMFFI_ExceptionValue : $result);
  let assemblyFormat = "$kind attr-dict `:` type($result)";
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
}

def TVMFFI_FuncOp : TVMFFI_Op<"func", [
  AffineScope, AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove,
  OpAsmOpInterface
]> {
  let summary = "A TVM FFI function with one C ABI return";
  let description = [{
    Defines a function that can be called through the TVM FFI interface.
    The function must return exactly one C ABI value. Pack multiple values
    into a tuple payload and use a None payload for functions with no value.

    Example:

    ```mlir
    tvm_ffi.func @add(%a: !tvm_ffi.int, %b: !tvm_ffi.int) -> !tvm_ffi.int {
      ...
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<UnitAttr>:$emit_tvm_ffi_abi,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion : $body);

  let builders = [OpBuilder<(ins
    "::llvm::StringRef":$name, "::mlir::FunctionType":$type,
    CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion();
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
    ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def TVMFFI_FunctionCallOp
    : TVMFFI_Op<"FunctionCall"> {
  let summary = "Call a TVM FFI function handle";
  let arguments = (ins TVMFFI_FunctionValue:$callee,
                       Variadic<AnyTVMFFISemanticType>:$arguments);
  let results = (outs AnyTVMFFISemanticType:$result);
  let assemblyFormat = "$callee `:` type($callee) `(` $arguments `)` attr-dict `:` functional-type($arguments, $result)";
}

def TVMFFI_FunctionGetGlobalOp
    : TVMFFI_Op<"FunctionGetGlobal", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Get a TVM FFI global function by name";
  let arguments = (ins StrAttr:$name);
  let results = (outs TVMFFI_FunctionValue:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
}

def TVMFFI_GetOp : TVMFFI_Op<"get", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extract a native value from a TVM FFI value";
  let arguments = (ins AnyTVMFFIGetOperandType:$operand);
  let results = (outs AnyTVMFFIGetResultType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

// Define the operand constraint before the launch operation that uses it.
def TVMFFI_KernelOperandType
    : AnyTypeOf<[AnyTVMFFISemanticType, F64, I1, I64],
                "TVM FFI value or native scalar accepted by a Triton launch">;

def TVMFFI_KernelLaunchOp
    : TVMFFI_Op<"kernel_launch", [
  AttrSizedOperandSegments, AllTypesMatch<[
    "gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX", "blockSizeY",
    "blockSizeZ"
  ]>
]> {
  let summary = "Launch a compiled kernel entry point";
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

let arguments = (ins SymbolRefAttr:$kernel,
                   I64:$gridSizeX,
                   I64:$gridSizeY,
                   I64:$gridSizeZ,
                   I64:$blockSizeX,
                   I64:$blockSizeY,
                   I64:$blockSizeZ,
                   Optional<I64>:$clusterSizeX,
                   Optional<I64>:$clusterSizeY,
                   Optional<I64>:$clusterSizeZ,
                   Optional<I32>:$dynamicSharedMemorySize,
                   Variadic<TVMFFI_KernelOperandType>:$kernelOperands,
                   TVMFFI_KernelSpecializationArrayAttr:$specializations);
let results = (outs TVMFFI_ResultType:$result);
let hasVerifier = 1;

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
}

include "trident/core/Dialect/TVMFFI/IR/Monad/TVMFFIOps.td"

def TVMFFI_ObjectDecRefOp : TVMFFI_Op<"ObjectDecRef"> {
  let summary = "Decrement a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ObjectIncRefOp : TVMFFI_Op<"ObjectIncRef"> {
  let summary = "Increment a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ReturnOp
    : TVMFFI_Op<"return", [HasParent<"FuncOp">, ReturnLike, Terminator]> {
  let summary = "Return explicit success and an Any payload from a C ABI function";
  let description = [{
    The function signature declares the successful C ABI payload type. Return
    carries a boxed success flag and an Any payload of that type on success,
    or None on failure. ABI lowering stores the payload only on success and
    preserves the runtime error on failure.
  }];
  let arguments = (ins TVMFFI_BoolType:$success, TVMFFI_AnyValue:$value);
  let assemblyFormat = "attr-dict $success `,` $value `:` type($success) `,` type($value)";
}

def TVMFFI_TensorCloneOp
    : TVMFFI_Op<"tensor.clone", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create an independent tensor copy preserving the layout";
  let arguments = (ins TVMFFI_TensorValue:$tensor);
  let results = (outs TVMFFI_TensorValue:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
}

def TVMFFI_TensorCopyOp
    : TVMFFI_Op<"tensor.copy_", [SameTypeOperands]> {
  let summary = "Copy tensor contents into an existing tensor in place";
  let arguments = (ins TVMFFI_TensorValue:$destination,
                       TVMFFI_TensorValue:$source);
  let assemblyFormat = "$destination `,` $source attr-dict `:` type($destination)";
}

def TVMFFI_TensorDTypeOp : TVMFFI_Op<"tensor.dtype", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack dtype fields from a runtime tensor";
  let arguments = (ins TVMFFI_TensorValue:$tensor);
  let results = (outs I8:$code, I8:$bits, I16:$lanes);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorDeviceOp : TVMFFI_Op<"tensor.device", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack device fields from a runtime tensor";
  let arguments = (ins TVMFFI_TensorValue:$tensor);
  let results = (outs I32:$device_type, I32:$device_index);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorDimOp : TVMFFI_Op<"tensor.dim", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read the rank of a runtime tensor";
  let arguments = (ins TVMFFI_TensorValue:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorLiteralOp
    : TVMFFI_Op<"tensor.literal", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Create an owned TVM FFI tensor from a dense literal";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs TVMFFI_TensorValue:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs,
                                        ::mlir::TypeRange rhs);
  }];
}

def TVMFFI_TensorSizeOp
    : TVMFFI_TensorIndexedMetadataOp<"tensor.size"> {
  let summary = "Read a runtime tensor dimension size";
}

def TVMFFI_TensorStorageOffsetOp
    : TVMFFI_Op<"tensor.storage_offset", [Pure,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read a runtime tensor storage offset";
  let arguments = (ins TVMFFI_TensorValue:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorStrideOp
    : TVMFFI_TensorIndexedMetadataOp<"tensor.stride"> {
  let summary = "Read a runtime tensor stride";
}

def TVMFFI_ToOp
    : TVMFFI_Op<"to", [Pure,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Convert a native scalar to a TVM FFI semantic value";
  let arguments = (ins AnyTypeOf<[F64, I1, I64]> : $value);
  let results = (outs AnyTypeOf<[TVMFFI_BoolType, TVMFFI_FloatType, TVMFFI_IntType]> : $result);
  let hasVerifier = 1;
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
}

def TVMFFI_TupleConstructOp : TVMFFI_Op<"tuple.construct", [
    RangedTypesMatchWith<"elements must match the tuple types", "result", "elements",
      "::llvm::cast<::trident::tvm_ffi::TupleType>($_self).getTypes()">
]> {
  let summary = "Pack elements into a semantic tuple";
  let arguments = (ins Variadic<AnyTVMFFISemanticType>:$elements);
  let results = (outs TVMFFI_TupleType:$result);
  let assemblyFormat = "`(` $elements `)` attr-dict `:` functional-type($elements, $result)";
}

def TVMFFI_TupleUnconstructOp : TVMFFI_Op<"tuple.unconstruct", [
    RangedTypesMatchWith<"results must match the tuple types", "container", "elements",
      "::llvm::cast<::trident::tvm_ffi::TupleType>($_self).getTypes()">
]> {
  let summary = "Unpack all semantic tuple elements in order";
  let arguments = (ins TVMFFI_TupleType:$container);
  let results = (outs Variadic<AnyTVMFFISemanticType>:$elements);
  let assemblyFormat = "$container attr-dict `:` functional-type($container, $elements)";
}

def TVMFFI_UpcastOp
    : TVMFFI_Op<"upcast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Widen a TVM FFI value, or preserve its type";
  let description = [{
    Identical source and result types are a no-op. Otherwise the result must
    be Any or a Union containing the source type.
  }];
  let arguments = (ins AnyTVMFFIAnyABIType : $value);
  let results = (outs AnyTVMFFIAnyABIType : $result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
  let hasFolder = 1;
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
