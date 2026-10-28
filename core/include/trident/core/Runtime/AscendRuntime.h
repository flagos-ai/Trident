//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_RUNTIME_ASCENDRUNTIME_H_
#define TRIDENT_CORE_RUNTIME_ASCENDRUNTIME_H_

#include <cstdint>

extern "C" {

enum TridentAscendArgumentType : uint8_t {
  TRIDENT_ASCEND_POINTER = 0,
  TRIDENT_ASCEND_I8 = 1,
  TRIDENT_ASCEND_I16 = 2,
  TRIDENT_ASCEND_I32 = 3,
  TRIDENT_ASCEND_I64 = 4,
  TRIDENT_ASCEND_F32 = 5,
  TRIDENT_ASCEND_F64 = 6,
};

int32_t tridentAscendLaunch(
    const uint8_t *binary, uint64_t binarySize, const char *kernelName,
    const char *mixMode, const uint8_t *argumentTypes, uint64_t argumentCount,
    const uint64_t *argumentValues, int32_t deviceIndex, void *stream,
    int64_t gridX, int64_t gridY, int64_t gridZ, int64_t workspaceSize,
    int64_t lockNum, int64_t lockInitValue, uint8_t hasFfts,
    uint8_t forceSimtOnly, uint8_t useV2Launch, uint8_t enableTaskQueue,
    uint8_t autoMapBlocks, int64_t sharedMemorySize,
    int64_t physicalBlockCount);

} // extern "C"

#endif // TRIDENT_CORE_RUNTIME_ASCENDRUNTIME_H_
