//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Runtime/AscendRuntime.h"
#include <ATen/ATen.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <runtime/runtime/rt.h>

namespace {

struct RegisteredKernel {
  void *binaryHandle = nullptr;
  std::vector<uint8_t> binary;
  std::unique_ptr<uint64_t> functionStub = std::make_unique<uint64_t>(0);
};

std::mutex registryMutex;
std::unordered_map<std::string, std::shared_ptr<RegisteredKernel>> kernels;
std::atomic<uint64_t> nextKernelId{0};

std::string kernelKey(const uint8_t *binary, uint64_t binarySize,
                      const char *kernelName, const char *mixMode,
                      int32_t device) {
  std::string key(reinterpret_cast<const char *>(binary), binarySize);
  key.push_back('\0');
  key.append(kernelName);
  key.push_back('\0');
  key.append(mixMode);
  key.push_back('\0');
  key.append(std::to_string(device));
  return key;
}

std::shared_ptr<RegisteredKernel>
registerKernel(const uint8_t *binary, uint64_t binarySize,
               const char *kernelName, const char *mixMode, int32_t device,
               rtError_t &status) {
  std::lock_guard<std::mutex> lock(registryMutex);
  status = rtSetDevice(device);
  if (status != RT_ERROR_NONE) {
    return {};
  }
  std::string const key =
      kernelKey(binary, binarySize, kernelName, mixMode, device);
  if (auto found = kernels.find(key); found != kernels.end()) {
    status = RT_ERROR_NONE;
    return found->second;
  }

  auto result = std::make_shared<RegisteredKernel>();
  result->binary.assign(binary, binary + binarySize);
  rtDevBinary_t devbin{};
  devbin.data = result->binary.data();
  devbin.length = binarySize;
  devbin.magic = std::strcmp(mixMode, "aiv") == 0
                     ? RT_DEV_BINARY_MAGIC_ELF_AIVEC
                     : RT_DEV_BINARY_MAGIC_ELF;
  devbin.version = 0;

  status = rtDevBinaryRegister(&devbin, &result->binaryHandle);
  if (status != RT_ERROR_NONE) {
    return {};
  }
  std::string const stubName =
      std::string(kernelName) + "_trident_" +
      std::to_string(nextKernelId.fetch_add(1, std::memory_order_relaxed));
  status = rtFunctionRegister(result->binaryHandle, result->functionStub.get(),
                              stubName.c_str(),
                              const_cast<char *>(kernelName), 0);
  if (status != RT_ERROR_NONE) {
    return {};
  }
  kernels.emplace(key, result);
  return result;
}

size_t alignTo(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
void appendAligned(std::vector<uint8_t> &buffer, T value, size_t alignment) {
  buffer.resize(alignTo(buffer.size(), alignment));
  size_t const offset = buffer.size();
  buffer.resize(offset + sizeof(T));
  std::memcpy(buffer.data() + offset, &value, sizeof(T));
}

rtError_t appendArgument(std::vector<uint8_t> &buffer, uint8_t type,
                         uint64_t value) {
  switch (type) {
  case TRIDENT_ASCEND_POINTER:
    appendAligned(buffer, reinterpret_cast<void *>(value), 8);
    return RT_ERROR_NONE;
  case TRIDENT_ASCEND_I8:
    appendAligned(buffer, static_cast<uint8_t>(value), 4);
    return RT_ERROR_NONE;
  case TRIDENT_ASCEND_I16:
    appendAligned(buffer, static_cast<uint16_t>(value), 4);
    return RT_ERROR_NONE;
  case TRIDENT_ASCEND_I32:
  case TRIDENT_ASCEND_F32:
    appendAligned(buffer, static_cast<uint32_t>(value), 4);
    return RT_ERROR_NONE;
  case TRIDENT_ASCEND_I64:
  case TRIDENT_ASCEND_F64:
    appendAligned(buffer, value, 8);
    return RT_ERROR_NONE;
  default:
    return static_cast<rtError_t>(1);
  }
}

struct LaunchState {
  std::shared_ptr<RegisteredKernel> kernel;
  std::vector<uint8_t> argumentTypes;
  std::vector<uint64_t> argumentValues;
  std::string kernelName;
  int64_t gridX = 1;
  int64_t gridY = 1;
  int64_t gridZ = 1;
  int64_t workspaceSize = 0;
  int64_t lockNum = 0;
  int64_t lockInitValue = 0;
  bool hasFfts = false;
  bool forceSimtOnly = false;
  bool useV2Launch = false;
  bool enableTaskQueue = true;
  bool autoMapBlocks = false;
  int64_t sharedMemorySize = 0;
  int64_t physicalBlockCount = 0;
  rtStream_t stream = nullptr;
  at::Tensor workspace;
  void *workspaceAddress = nullptr;
};

rtError_t launch(const std::shared_ptr<LaunchState> &state) {
  uint64_t const rawBlockCount =
      static_cast<uint64_t>(state->gridX) *
      static_cast<uint64_t>(state->gridY) *
      static_cast<uint64_t>(state->gridZ);
  uint32_t blockCount = static_cast<uint32_t>(rawBlockCount);
  if (state->autoMapBlocks && state->physicalBlockCount > 0) {
    blockCount = std::min(
        blockCount, static_cast<uint32_t>(state->physicalBlockCount));
  }

  void *fftsAddress = nullptr;
  if (state->hasFfts) {
    uint32_t fftsLength = 0;
    rtError_t status =
        rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&fftsAddress),
                         &fftsLength);
    if (status != RT_ERROR_NONE) {
      return status;
    }
  }

  at::Tensor lockStorage;
  void *lockAddress = nullptr;
  if (state->lockNum > 0) {
    uint64_t const lockBytes =
        static_cast<uint64_t>(state->lockNum) * sizeof(int64_t);
    lockStorage =
        at_npu::native::allocate_workspace(lockBytes, state->stream);
    lockAddress = const_cast<void *>(lockStorage.storage().data());
    std::vector<int64_t> initialValues(
        static_cast<size_t>(state->lockNum), state->lockInitValue);
    rtError_t status =
        rtMemcpy(lockAddress, lockBytes, initialValues.data(), lockBytes,
                 RT_MEMCPY_HOST_TO_DEVICE);
    if (status != RT_ERROR_NONE) {
      return status;
    }
  }

  std::vector<uint8_t> packed;
  size_t structAlignment = 4;
  if (state->hasFfts) {
    appendAligned(packed, fftsAddress, 8);
    structAlignment = 8;
  }
  if (!state->forceSimtOnly) {
    appendAligned(packed, lockAddress, 8);
    appendAligned(packed, state->workspaceAddress, 8);
    structAlignment = 8;
  }
  for (size_t index = 0; index < state->argumentTypes.size(); ++index) {
    if (state->argumentTypes[index] == TRIDENT_ASCEND_POINTER ||
        state->argumentTypes[index] == TRIDENT_ASCEND_I64 ||
        state->argumentTypes[index] == TRIDENT_ASCEND_F64) {
      structAlignment = 8;
    }
    rtError_t status = appendArgument(
        packed, state->argumentTypes[index], state->argumentValues[index]);
    if (status != RT_ERROR_NONE) {
      return status;
    }
  }
  appendAligned(packed, static_cast<int32_t>(state->gridX), 4);
  appendAligned(packed, static_cast<int32_t>(state->gridY), 4);
  appendAligned(packed, static_cast<int32_t>(state->gridZ), 4);
  packed.resize(alignTo(packed.size(), structAlignment));

  void *function = state->kernel->functionStub.get();
  if (state->useV2Launch) {
    rtArgsEx_t argsInfo{};
    argsInfo.args = packed.data();
    argsInfo.argsSize = packed.size();
    rtTaskCfgInfo_t config{};
    config.localMemorySize = state->sharedMemorySize;
    return rtKernelLaunchWithFlagV2(function, blockCount, &argsInfo, nullptr,
                                    state->stream, 0, &config);
  }
  return rtKernelLaunch(function, blockCount, packed.data(), packed.size(),
                        nullptr, state->stream);
}

} // namespace

extern "C" int32_t tridentAscendLaunch(
    const uint8_t *binary, uint64_t binarySize, const char *kernelName,
    const char *mixMode, const uint8_t *argumentTypes, uint64_t argumentCount,
    const uint64_t *argumentValues, int32_t deviceIndex, void *stream,
    int64_t gridX, int64_t gridY, int64_t gridZ, int64_t workspaceSize,
    int64_t lockNum, int64_t lockInitValue, uint8_t hasFfts,
    uint8_t forceSimtOnly, uint8_t useV2Launch, uint8_t enableTaskQueue,
    uint8_t autoMapBlocks, int64_t sharedMemorySize,
    int64_t physicalBlockCount) {
  if (!binary || !kernelName || !mixMode || !argumentTypes ||
      (!argumentValues && argumentCount != 0) || !stream || gridX <= 0 ||
      gridY <= 0 || gridZ <= 0) {
    return 1;
  }
  rtError_t status = RT_ERROR_NONE;
  auto kernel = registerKernel(binary, binarySize, kernelName, mixMode,
                               deviceIndex, status);
  if (!kernel) {
    return static_cast<int32_t>(status);
  }

  auto state = std::make_shared<LaunchState>();
  state->kernel = std::move(kernel);
  state->argumentTypes.assign(argumentTypes, argumentTypes + argumentCount);
  state->argumentValues.assign(argumentValues, argumentValues + argumentCount);
  state->kernelName = kernelName;
  state->gridX = gridX;
  state->gridY = gridY;
  state->gridZ = gridZ;
  state->workspaceSize = workspaceSize;
  state->lockNum = lockNum;
  state->lockInitValue = lockInitValue;
  state->hasFfts = hasFfts;
  state->forceSimtOnly = forceSimtOnly;
  state->useV2Launch = useV2Launch;
  state->enableTaskQueue = enableTaskQueue;
  state->autoMapBlocks = autoMapBlocks;
  state->sharedMemorySize = sharedMemorySize;
  state->physicalBlockCount = physicalBlockCount;
  state->stream = static_cast<rtStream_t>(stream);
  if (workspaceSize > 0) {
    uint64_t const blockCount =
        static_cast<uint64_t>(gridX) * static_cast<uint64_t>(gridY) *
        static_cast<uint64_t>(gridZ);
    uint64_t const totalSize =
        static_cast<uint64_t>(workspaceSize) * blockCount;
    state->workspace = at::empty(
        totalSize,
        at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte));
    state->workspaceAddress =
        const_cast<void *>(state->workspace.storage().data());
  }

  if (!state->enableTaskQueue) {
    status = launch(state);
    if (status == RT_ERROR_NONE) {
      status = rtStreamSynchronize(state->stream);
    }
    return static_cast<int32_t>(status);
  }
  at_npu::native::OpCommand command;
  command.Name(state->kernelName)
      .SetCustomHandler([state]() -> rtError_t { return launch(state); })
      .Run();
  return 0;
}
