# SPDX-License-Identifier: MIT

# Part of the Trident project, under the MIT License.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TVMFFIArithTransforms)
add_public_tablegen_target(MLIRTVMFFIArithTransformsPassIncGen)
