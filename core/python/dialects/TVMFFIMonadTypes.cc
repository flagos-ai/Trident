//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFIMonadTypes.h"
#include "trident-c/core/Dialects.h"
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Bindings/Python/Diagnostics.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // IWYU pragma: keep

namespace nb = nanobind;
namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::monad {
struct ResultType final : PyConcreteType<ResultType> {
  using Base::Base;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIMonadResultTypeGetTypeID;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIMonadResultType;
  static inline const MlirStringRef name = mlirTVMFFIMonadResultTypeGetName();
  static constexpr const char *pyClassName = "ResultType";

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const PyType &payload,
           DefaultingPyMlirContext context) -> ResultType {
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIMonadResultTypeGetChecked(
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
      const MlirType payload = mlirTVMFFIMonadResultTypeGetPayloadType(type);
      return nb::cast(PyType(type.getContext(), payload).maybeDownCast());
    });
  }
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::monad
NB_MODULE(_tvm_ffi_monad, module) {
  module.doc() = "TVM FFI monad dialect bindings";
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi::monad;
  ResultType::bind(module);
  module.def(
      "register_dialect",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirDialectHandle const handle =
            mlirGetDialectHandle__tvm_ffi_monad__();
        mlirDialectHandleRegisterDialect(handle, context.get()->get());
        if (load) {
          mlirDialectHandleLoadDialect(handle, context.get()->get());
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
