//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

namespace {

void reportHipError(const char *expression, hipError_t status) {
  if (status == hipSuccess) {
    return;
  }
  const char *message = hipGetErrorString(status);
  std::fprintf(stderr, "Trident Hcu runtime: %s failed with code %d: %s\n",
               expression, static_cast<int>(status),
               message ? message : "<unknown>");
}

} // namespace

extern "C" hipModule_t mgpuModuleLoad(void *data, std::size_t) {
  hipModule_t module = nullptr;
  const hipError_t status = hipModuleLoadData(&module, data);
  reportHipError("hipModuleLoadData", status);
  return status == hipSuccess ? module : nullptr;
}

extern "C" void mgpuModuleUnload(hipModule_t module) {
  if (module != nullptr) {
    reportHipError("hipModuleUnload", hipModuleUnload(module));
  }
}

extern "C" hipFunction_t mgpuModuleGetFunction(hipModule_t module,
                                               const char *name) {
  hipFunction_t function = nullptr;
  const hipError_t status = hipModuleGetFunction(&function, module, name);
  reportHipError("hipModuleGetFunction", status);
  return status == hipSuccess ? function : nullptr;
}

extern "C" void mgpuLaunchKernel(hipFunction_t function, std::intptr_t gridX,
                                 std::intptr_t gridY, std::intptr_t gridZ,
                                 std::intptr_t blockX, std::intptr_t blockY,
                                 std::intptr_t blockZ, std::int32_t sharedMemory,
                                 hipStream_t stream, void **parameters,
                                 void **extra, std::size_t) {
  reportHipError(
      "hipModuleLaunchKernel",
      hipModuleLaunchKernel(
          function, static_cast<unsigned int>(gridX),
          static_cast<unsigned int>(gridY), static_cast<unsigned int>(gridZ),
          static_cast<unsigned int>(blockX),
          static_cast<unsigned int>(blockY),
          static_cast<unsigned int>(blockZ),
          static_cast<unsigned int>(sharedMemory), stream, parameters, extra));
}
