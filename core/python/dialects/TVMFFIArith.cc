// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT

#include "trident-c/core/Dialects.h"
#include <mlir-c/IR.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <nanobind/nanobind.h>
namespace nb = nanobind;
NB_MODULE(_tvm_ffi_arith, module) {
  module.doc() = "TVM FFI arithmetic dialect bindings";
  module.def(
      "register_dialect",
      [](mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::DefaultingPyMlirContext
             context,
         bool load) {
        MlirDialectHandle const handle =
            mlirGetDialectHandle__tvm_ffi_arith__();
        mlirDialectHandleRegisterDialect(handle, context.get()->get());
        if (load) {
          mlirDialectHandleLoadDialect(handle, context.get()->get());
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
