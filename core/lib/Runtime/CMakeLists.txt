if(NOT TRIDENT_ASCEND_BACKEND)
  return()
endif()

if("$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  message(
    FATAL_ERROR
      "FLAGTREE_BACKEND=ascend requires ASCEND_HOME_PATH; source the Ascend toolkit set_env.sh")
endif()
set(TRIDENT_ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
  OUTPUT_VARIABLE TRIDENT_TORCH_NPU_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TRIDENT_TORCH_NPU_QUERY_RESULT
  ERROR_VARIABLE TRIDENT_TORCH_NPU_QUERY_ERROR)
if(NOT TRIDENT_TORCH_NPU_QUERY_RESULT STREQUAL "0")
  message(
    FATAL_ERROR
      "FLAGTREE_BACKEND=ascend requires torch_npu:\n${TRIDENT_TORCH_NPU_QUERY_ERROR}")
endif()

add_library(TridentAscendRuntime SHARED AscendRuntime.cc)
separate_arguments(_trident_torch_cxx_flags NATIVE_COMMAND
                   "${TORCH_CXX_FLAGS}")
target_compile_options(TridentAscendRuntime
                       PRIVATE ${_trident_torch_cxx_flags})
target_include_directories(
  TridentAscendRuntime
  PRIVATE
    "${TRIDENT_ASCEND_HOME}/include"
    "${TRIDENT_ASCEND_HOME}/include/experiment"
    "${TRIDENT_ASCEND_HOME}/include/experiment/msprof"
    "${TRIDENT_ASCEND_HOME}/pkg_inc"
    "${TRIDENT_ASCEND_HOME}/pkg_inc/profiling"
    "${TRIDENT_TORCH_NPU_ROOT}/include")
target_link_directories(
  TridentAscendRuntime
  PRIVATE "${TRIDENT_ASCEND_HOME}/lib64" "${TRIDENT_TORCH_NPU_ROOT}/lib")
target_link_libraries(
  TridentAscendRuntime
  PRIVATE ${TORCH_LIBRARIES} runtime ascendcl torch_npu)
set_target_properties(
  TridentAscendRuntime
  PROPERTIES INSTALL_RPATH
             "$ORIGIN;${TRIDENT_TORCH_NPU_ROOT}/lib;${TRIDENT_ASCEND_HOME}/lib64")

install(
  TARGETS TridentAscendRuntime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
