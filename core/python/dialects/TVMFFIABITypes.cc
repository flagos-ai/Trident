//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFIABITypes.h" // NOLINT(misc-include-cleaner)
#include "trident-c/core/Dialects.h"
#include <cstdint> // NOLINT(misc-include-cleaner)
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Bindings/Python/Diagnostics.h> // NOLINT(misc-include-cleaner)
#include <mlir/Bindings/Python/IRCore.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // IWYU pragma: keep
#include <vector>                // NOLINT(misc-include-cleaner)

namespace nb = nanobind;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::abi {

template <typename DerivedTy> struct SimpleType : PyConcreteType<DerivedTy> {
  using Base = PyConcreteType<DerivedTy>;
  using GetFunctionTy = MlirType (*)(MlirContext);
  using Base::Base;

  static void bindDerived(typename Base::ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](DefaultingPyMlirContext context) -> DerivedTy {
          return DerivedTy(context->getRef(),
                           DerivedTy::getFunction(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
    typeClass.def_static(
        "isinstance",
        [](const PyType &type) -> bool { return DerivedTy::isaFunction(type); },
        nb::arg("other_type"));
  }
};

#define DEFINE_GENERIC_TYPE(Name)                                              \
  struct Name##Type final : SimpleType<Name##Type> {                           \
    using SimpleType<Name##Type>::SimpleType;                                  \
    static constexpr GetFunctionTy getFunction = mlirTVMFFIABI##Name##TypeGet; \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction =                   \
        mlirTVMFFIABI##Name##TypeGetTypeID;                                    \
    static constexpr IsAFunctionTy isaFunction =                               \
        mlirTypeIsATVMFFIABI##Name##Type;                                      \
    static inline const MlirStringRef name =                                   \
        mlirTVMFFIABI##Name##TypeGetName();                                    \
    static constexpr const char *pyClassName = #Name "Type";                   \
  };

DEFINE_GENERIC_TYPE(Any)
DEFINE_GENERIC_TYPE(Array)
DEFINE_GENERIC_TYPE(Bool)
DEFINE_GENERIC_TYPE(DType)
DEFINE_GENERIC_TYPE(Device)
DEFINE_GENERIC_TYPE(Exception)
DEFINE_GENERIC_TYPE(Float)
DEFINE_GENERIC_TYPE(Function)
DEFINE_GENERIC_TYPE(Int)
DEFINE_GENERIC_TYPE(None)
DEFINE_GENERIC_TYPE(Object)
DEFINE_GENERIC_TYPE(PackArguments)
DEFINE_GENERIC_TYPE(RawStr)
DEFINE_GENERIC_TYPE(SmallStr)
DEFINE_GENERIC_TYPE(Str)
DEFINE_GENERIC_TYPE(Tensor)

#undef DEFINE_GENERIC_TYPE

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::abi

NB_MODULE(_tvm_ffi_abi, module) {
  module.doc() = "TVM FFI ABI dialect bindings";
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::abi;
  AnyType::bind(module);
  ArrayType::bind(module);
  BoolType::bind(module);
  DTypeType::bind(module);
  DeviceType::bind(module);
  ExceptionType::bind(module);
  FloatType::bind(module);
  FunctionType::bind(module);
  IntType::bind(module);
  NoneType::bind(module);
  ObjectType::bind(module);
  PackArgumentsType::bind(module);
  RawStrType::bind(module);
  SmallStrType::bind(module);
  StrType::bind(module);
  TensorType::bind(module);
  module.def(
      "register_dialect",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirDialectHandle const handle = mlirGetDialectHandle__tvm_ffi_abi__();
        mlirDialectHandleRegisterDialect(handle, context.get()->get());
        if (load) {
          mlirDialectHandleLoadDialect(handle, context.get()->get());
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
