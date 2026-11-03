//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFITypes.h"
#include <cstdint>
#include <mlir-c/IR.h>
#include <mlir/Bindings/Python/Diagnostics.h>
#include <mlir/Bindings/Python/NanobindAdaptors.h>
#include <vector>

namespace nb = nanobind;

NB_MODULE(_tvm_ffi, module) {
  module.doc() = "TVM FFI dialect type bindings";
  using mlir::python::CollectDiagnosticsToStringScope;
  using mlir::python::nanobind_adaptors::mlir_type_subclass;

#define BIND_SIMPLE_TYPE(Name)                                                 \
  mlir_type_subclass(module, #Name "Type", mlirTypeIsATVMFFI##Name##Type,     \
                     mlirTVMFFI##Name##TypeGetTypeID)                          \
      .def_classmethod(                                                        \
          "get",                                                               \
          [](const nb::object &cls, MlirContext context) {                     \
            return cls(mlirTVMFFI##Name##TypeGet(context));                    \
          },                                                                   \
          nb::arg("cls"), nb::kw_only(),                                       \
          nb::arg("context").none() = nb::none());

  BIND_SIMPLE_TYPE(Any)
  BIND_SIMPLE_TYPE(Array)
  BIND_SIMPLE_TYPE(Bool)
  BIND_SIMPLE_TYPE(DType)
  BIND_SIMPLE_TYPE(Device)
  BIND_SIMPLE_TYPE(Exception)
  BIND_SIMPLE_TYPE(Float)
  BIND_SIMPLE_TYPE(Function)
  BIND_SIMPLE_TYPE(Int)
  BIND_SIMPLE_TYPE(None)
  BIND_SIMPLE_TYPE(Object)
  BIND_SIMPLE_TYPE(RawStr)
  BIND_SIMPLE_TYPE(SmallStr)
  BIND_SIMPLE_TYPE(Str)
  BIND_SIMPLE_TYPE(Tensor)

#undef BIND_SIMPLE_TYPE

  mlir_type_subclass(module, "UnionType", mlirTypeIsATVMFFIUnionType,
                     mlirTVMFFIUnionTypeGetTypeID)
      .def_classmethod(
          "get",
          [](const nb::object &cls, const std::vector<MlirType> &types,
             MlirContext context) {
            CollectDiagnosticsToStringScope diagnostics(context);
            const MlirType type = mlirTVMFFIUnionTypeGetChecked(
                mlirLocationUnknownGet(context),
                static_cast<intptr_t>(types.size()), types.data());
            if (mlirTypeIsNull(type))
              throw nb::value_error(diagnostics.takeMessage().c_str());
            return cls(type);
          },
          nb::arg("cls"), nb::arg("types"), nb::kw_only(),
          nb::arg("context").none() = nb::none())
      .def_property_readonly("types", [](MlirType type) {
        nb::list members;
        const intptr_t size = mlirTVMFFIUnionTypeGetNumTypes(type);
        for (intptr_t index = 0; index < size; ++index)
          members.append(mlirTVMFFIUnionTypeGetType(type, index));
        return nb::tuple(members);
      });
}
