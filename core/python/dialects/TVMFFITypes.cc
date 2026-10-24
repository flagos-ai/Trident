//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFITypes.h"
#include <cstdint>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Bindings/Python/Diagnostics.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // IWYU pragma: keep
#include <vector>

namespace nb = nanobind;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi {

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
    static constexpr GetFunctionTy getFunction = mlirTVMFFI##Name##TypeGet;    \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction =                   \
        mlirTVMFFI##Name##TypeGetTypeID;                                       \
    static constexpr IsAFunctionTy isaFunction =                               \
        mlirTypeIsATVMFFI##Name##Type;                                         \
    static inline const MlirStringRef name = mlirTVMFFI##Name##TypeGetName();  \
    static constexpr const char *pyClassName = #Name "Type";                   \
  };

DEFINE_GENERIC_TYPE(Any)
DEFINE_GENERIC_TYPE(Bool)
DEFINE_GENERIC_TYPE(CAny)
DEFINE_GENERIC_TYPE(CArray)
DEFINE_GENERIC_TYPE(CException)
DEFINE_GENERIC_TYPE(CFunction)
DEFINE_GENERIC_TYPE(CStr)
DEFINE_GENERIC_TYPE(CTensor)
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

struct ArrayType final : PyConcreteType<ArrayType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIArrayTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIArrayType;
  static inline const MlirStringRef name = mlirTVMFFIArrayTypeGetName();
  static constexpr const char *pyClassName = "ArrayType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const PyType &element,
           DefaultingPyMlirContext context) -> ArrayType {
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIArrayTypeGetChecked(
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
      const MlirType element = mlirTVMFFIArrayTypeGetElementType(type);
      return nb::cast(PyType(type.getContext(), element).maybeDownCast());
    });
  }
};

struct ResultType final : PyConcreteType<ResultType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIResultTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIResultType;
  static inline const MlirStringRef name = mlirTVMFFIResultTypeGetName();
  static constexpr const char *pyClassName = "ResultType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const PyType &payload,
           DefaultingPyMlirContext context) -> ResultType {
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIResultTypeGetChecked(
              mlirLocationUnknownGet(context.get()->get()),
              static_cast<MlirType>(payload));
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(diagnostics.takeMessage().c_str());
          }
          return ResultType(context->getRef(), type);
        },
        nb::arg("payload_type"), nb::kw_only(),
        nb::arg("context").none() = nb::none());
    typeClass.def_prop_ro("payload_type", [](ResultType &type) -> nb::object {
      const MlirType payload = mlirTVMFFIResultTypeGetPayloadType(type);
      return nb::cast(PyType(type.getContext(), payload).maybeDownCast());
    });
  }
};

struct TupleType final : PyConcreteType<TupleType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFITupleTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFITupleType;
  static inline const MlirStringRef name = mlirTVMFFITupleTypeGetName();
  static constexpr const char *pyClassName = "TupleType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const std::vector<PyType> &types,
           DefaultingPyMlirContext context) -> TupleType {
          std::vector<MlirType> members(types.begin(), types.end());
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFITupleTypeGetChecked(
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
      const intptr_t size = mlirTVMFFITupleTypeGetNumTypes(type);
      for (intptr_t index = 0; index < size; ++index) {
        members.append(
            PyType(type.getContext(), mlirTVMFFITupleTypeGetType(type, index))
                .maybeDownCast());
      }
      return nb::tuple(members);
    });
  }
};

struct UnionType final : PyConcreteType<UnionType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIUnionTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIUnionType;
  static inline const MlirStringRef name = mlirTVMFFIUnionTypeGetName();
  static constexpr const char *pyClassName = "UnionType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const std::vector<PyType> &types,
           DefaultingPyMlirContext context) -> UnionType {
          std::vector<MlirType> members(types.begin(), types.end());
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIUnionTypeGetChecked(
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
      const intptr_t size = mlirTVMFFIUnionTypeGetNumTypes(type);
      for (intptr_t index = 0; index < size; ++index) {
        members.append(
            PyType(type.getContext(), mlirTVMFFIUnionTypeGetType(type, index))
                .maybeDownCast());
      }
      return nb::tuple(members);
    });
  }
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi

NB_MODULE(_tvm_ffi, module) {
  module.doc() = "TVM FFI dialect type bindings";
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi;
  AnyType::bind(module);
  ArrayType::bind(module);
  BoolType::bind(module);
  CAnyType::bind(module);
  CArrayType::bind(module);
  CExceptionType::bind(module);
  CFunctionType::bind(module);
  CStrType::bind(module);
  CTensorType::bind(module);
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
  ResultType::bind(module);
  SmallStrType::bind(module);
  StrType::bind(module);
  TensorType::bind(module);
  TupleType::bind(module);
  UnionType::bind(module);
}
