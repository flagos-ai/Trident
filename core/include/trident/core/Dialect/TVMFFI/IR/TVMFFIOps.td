//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.td"

class TVMFFI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVMFFI_Dialect, mnemonic, traits>;

class TVMFFI_TensorIndexedMetadataOp<string mnemonic>
    : TVMFFI_Op<mnemonic, [Pure,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins TVMFFI_TensorType : $tensor, I64 : $index);
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

def TVMFFI_CallOp : TVMFFI_Op<"call", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Call a TVM FFI function defined in the current module";
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyTVMFFIAnyABIType>:$operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs AnyTVMFFIAnyABIType:$result);
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` "
                       "functional-type($operands, $result)";
}

class TVMFFI_ConstantOp<string mnemonic, Type resultType, Attr valueType>
    : TVMFFI_Op<mnemonic, [
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TVMFFI_ObjectOwnershipOpInterface]> {
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
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create a TVM FFI None constant";
  let results = (outs TVMFFI_NoneType:$result);
  let assemblyFormat = "attr-dict";
}

def TVMFFI_ConstantRawStrOp
    : TVMFFI_ConstantOp<"constant.raw_str", TVMFFI_RawStrType, StrAttr> {
  let summary = "Create a TVM FFI raw string constant";
}

// Deliberately not Pure: a failed check must trap even if the result is unused.
def TVMFFI_DowncastOp
    : TVMFFI_Op<"downcast", [DeclareOpInterfaceMethods<CastOpInterface>,
        DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                                  ["getObjectResultOwnership"]>]> {
  let summary = "Check and narrow an any or union TVM FFI value";
  let arguments = (ins AnyTVMFFICastResultType : $value);
  let results = (outs AnyTVMFFIAnyABIType : $result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
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
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create a Trident FFI exception object";
  let arguments = (ins StrAttr : $kind);
  let results = (outs TVMFFI_ExceptionType : $result);
  let assemblyFormat = "$kind attr-dict `:` type($result)";
}

def TVMFFI_FuncOp : TVMFFI_Op<"func", [
  AffineScope, AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove,
  OpAsmOpInterface
]> {
  let summary = "A TVM FFI function with one monadic Result return";
  let description = [{
    Defines a function that can be called through the TVM FFI interface.
    The function must return exactly one monadic Result. Pack multiple values
    into a tuple payload and use a None payload for functions with no value.

    Example:

    ```mlir
    tvm_ffi.func @add(%a: !tvm_ffi.int, %b: !tvm_ffi.int) -> !tvm_ffi.monad.result<!tvm_ffi.int> {
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
    : TVMFFI_Op<"FunctionCall", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Call a TVM FFI function handle";
  let arguments = (ins TVMFFI_ObjectType:$callee,
                       Variadic<AnyTVMFFISemanticType>:$arguments);
  let results = (outs AnyTVMFFISemanticType:$result);
  let assemblyFormat = "$callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $result)";
}

def TVMFFI_FunctionGetGlobalOp
    : TVMFFI_Op<"FunctionGetGlobal", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Get a TVM FFI global function by name";
  let arguments = (ins StrAttr:$name);
  let results = (outs TVMFFI_ObjectType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
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

def TVMFFI_MonadFailOp : TVMFFI_Op<"monad.fail", [Pure]> {
  let summary = "Construct failed Results without changing the runtime error";
  let results = (outs Variadic<TVMFFI_ResultType>:$outputs);
  let assemblyFormat = "attr-dict `:` type($outputs)";
  let hasVerifier = 1;
}

def TVMFFI_MonadPureOp : TVMFFI_Op<"monad.pure", [Pure]> {
  let summary = "Wrap payloads in successful Results";
  let description = [{
    Transfers payload references by default. Frontend builders may attach the
    tvm_ffi.retain UnitAttr to borrow payloads instead: cleanup
    materialization retains each occurrence after semantic container erasure.
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
}

def TVMFFI_MonadYieldOp
    : TVMFFI_Op<"monad.yield", [Terminator]> {
  let summary = "Yield Results from a bind continuation";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

def TVMFFI_ObjectDecRefOp : TVMFFI_Op<"ObjectDecRef", [
    DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                              ["getObjectOperandOwnership"]>]> {
  let summary = "Decrement a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ObjectIncRefOp : TVMFFI_Op<"ObjectIncRef", [
    DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                              ["getObjectOperandOwnership"]>]> {
  let summary = "Increment a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ReturnOp
    : TVMFFI_Op<"return", [HasParent<"FuncOp">, ReturnLike, Terminator]> {
  let summary = "Return successful Result payloads or propagate failure to the ABI";
  let description = [{
    Returns monadic Results from a TVM FFI function. The operand count and
    types must match the function signature result types.

    Example:

    ```mlir
    tvm_ffi.return %result : !tvm_ffi.monad.result<!tvm_ffi.int>
    ```
  }];

  let arguments = (ins Variadic<TVMFFI_ResultType> : $operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def TVMFFI_TensorCloneOp
    : TVMFFI_Op<"tensor.clone", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create an independent tensor copy preserving the layout";
  let arguments = (ins TVMFFI_TensorType:$tensor);
  let results = (outs TVMFFI_TensorType:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";
}

def TVMFFI_TensorCopyOp
    : TVMFFI_Op<"tensor.copy_", []> {
  let summary = "Copy tensor contents into an existing tensor in place";
  let arguments = (ins TVMFFI_TensorType:$destination,
                       TVMFFI_TensorType:$source);
  let assemblyFormat = "$destination `,` $source attr-dict `:` type($destination)";
}

def TVMFFI_TensorDTypeOp : TVMFFI_Op<"tensor.dtype", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack dtype fields from a runtime tensor";
  let arguments = (ins TVMFFI_TensorType:$tensor);
  let results = (outs I8:$code, I8:$bits, I16:$lanes);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorDeviceOp : TVMFFI_Op<"tensor.device", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read DLPack device fields from a runtime tensor";
  let arguments = (ins TVMFFI_TensorType:$tensor);
  let results = (outs I32:$device_type, I32:$device_index);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorDimOp : TVMFFI_Op<"tensor.dim", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read the rank of a runtime tensor";
  let arguments = (ins TVMFFI_TensorType:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorLiteralOp
    : TVMFFI_Op<"tensor.literal", [
        DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create an owned TVM FFI tensor from a dense literal";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs TVMFFI_TensorType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TVMFFI_TensorSizeOp
    : TVMFFI_TensorIndexedMetadataOp<"tensor.size"> {
  let summary = "Read a runtime tensor dimension size";
}

def TVMFFI_TensorStorageOffsetOp
    : TVMFFI_Op<"tensor.storage_offset", [Pure,
        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Read a runtime tensor storage offset";
  let arguments = (ins TVMFFI_TensorType:$tensor);
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
    : TVMFFI_Op<"upcast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>,
        DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                                  ["getObjectResultOwnership"]>]> {
  let summary = "Widen a TVM FFI value into an any or union ABI type";
  let arguments = (ins AnyTVMFFIAnyABIType : $value);
  let results = (outs AnyTVMFFICastResultType : $result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
