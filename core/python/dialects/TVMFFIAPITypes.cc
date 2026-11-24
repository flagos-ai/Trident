//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFIAPITypes.h"
#include "trident-c/core/Dialects.h"
#include <cstdint>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Bindings/Python/Diagnostics.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // IWYU pragma: keep
#include <vector>

namespace nb = nanobind;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::api {

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
    static constexpr GetFunctionTy getFunction = mlirTVMFFIAPI##Name##TypeGet; \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction =                   \
        mlirTVMFFIAPI##Name##TypeGetTypeID;                                    \
    static constexpr IsAFunctionTy isaFunction =                               \
        mlirTypeIsATVMFFIAPI##Name##Type;                                      \
    static inline const MlirStringRef name =                                   \
        mlirTVMFFIAPI##Name##TypeGetName();                                    \
    static constexpr const char *pyClassName = #Name "Type";                   \
  };

DEFINE_GENERIC_TYPE(Any)
DEFINE_GENERIC_TYPE(Exception)
DEFINE_GENERIC_TYPE(Function)
DEFINE_GENERIC_TYPE(Str)
DEFINE_GENERIC_TYPE(Tensor)

#undef DEFINE_GENERIC_TYPE

struct ArrayType final : PyConcreteType<ArrayType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIAPIArrayTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIAPIArrayType;
  static inline const MlirStringRef name = mlirTVMFFIAPIArrayTypeGetName();
  static constexpr const char *pyClassName = "ArrayType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const PyType &element,
           DefaultingPyMlirContext context) -> ArrayType {
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIAPIArrayTypeGetChecked(
              mlirLocationUnknownGet(context.get()->get()),
              static_cast<MlirType>(element));
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(diagnostics.takeMessage().c_str());
          }
          return ArrayType(context->getRef(), type);
        },
        nb::arg("element_type"), nb::kw_only(),
        nb::arg("context").none() = nb::none());
    typeClass.def_prop_ro("element_type", [](ArrayType &type) -> nb::object {
      const MlirType element = mlirTVMFFIAPIArrayTypeGetElementType(type);
      return nb::cast(PyType(type.getContext(), element).maybeDownCast());
    });
  }
};

struct TupleType final : PyConcreteType<TupleType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIAPITupleTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIAPITupleType;
  static inline const MlirStringRef name = mlirTVMFFIAPITupleTypeGetName();
  static constexpr const char *pyClassName = "TupleType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const std::vector<PyType> &types,
           DefaultingPyMlirContext context) -> TupleType {
          std::vector<MlirType> members(types.begin(), types.end());
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIAPITupleTypeGetChecked(
              mlirLocationUnknownGet(context.get()->get()),
              static_cast<intptr_t>(members.size()), members.data());
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(diagnostics.takeMessage().c_str());
          }
          return TupleType(context->getRef(), type);
        },
        nb::arg("types"), nb::kw_only(),
        nb::arg("context").none() = nb::none());
    typeClass.def_static(
        "isinstance",
        [](const PyType &type) -> bool { return isaFunction(type); },
        nb::arg("other_type"));
    typeClass.def_prop_ro("types", [](TupleType &type) -> nb::tuple {
      nb::list members;
      const intptr_t size = mlirTVMFFIAPITupleTypeGetNumTypes(type);
      for (intptr_t index = 0; index < size; ++index) {
        members.append(PyType(type.getContext(),
                              mlirTVMFFIAPITupleTypeGetType(type, index))
                           .maybeDownCast());
      }
      return nb::tuple(members);
    });
  }
};

struct UnionType final : PyConcreteType<UnionType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIAPIUnionTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIAPIUnionType;
  static inline const MlirStringRef name = mlirTVMFFIAPIUnionTypeGetName();
  static constexpr const char *pyClassName = "UnionType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const std::vector<PyType> &types,
           DefaultingPyMlirContext context) -> UnionType {
          std::vector<MlirType> members(types.begin(), types.end());
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIAPIUnionTypeGetChecked(
              mlirLocationUnknownGet(context.get()->get()),
              static_cast<intptr_t>(members.size()), members.data());
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(diagnostics.takeMessage().c_str());
          }
          return UnionType(context->getRef(), type);
        },
        nb::arg("types"), nb::kw_only(),
        nb::arg("context").none() = nb::none());
    typeClass.def_static(
        "isinstance",
        [](const PyType &type) -> bool { return isaFunction(type); },
        nb::arg("other_type"));
    typeClass.def_prop_ro("types", [](UnionType &type) -> nb::tuple {
      nb::list members;
      const intptr_t size = mlirTVMFFIAPIUnionTypeGetNumTypes(type);
      for (intptr_t index = 0; index < size; ++index) {
        members.append(PyType(type.getContext(),
                              mlirTVMFFIAPIUnionTypeGetType(type, index))
                           .maybeDownCast());
      }
      return nb::tuple(members);
    });
  }
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::api

NB_MODULE(_tvm_ffi_api, module) {
  module.doc() = "TVM FFI API dialect bindings";
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::api;
  AnyType::bind(module);
  ArrayType::bind(module);
  ExceptionType::bind(module);
  FunctionType::bind(module);
  StrType::bind(module);
  TensorType::bind(module);
  TupleType::bind(module);
  UnionType::bind(module);
  module.def(
      "register_dialect",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirDialectHandle const handle = mlirGetDialectHandle__tvm_ffi_api__();
        mlirDialectHandleRegisterDialect(handle, context.get()->get());
        if (load) {
          mlirDialectHandleLoadDialect(handle, context.get()->get());
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
