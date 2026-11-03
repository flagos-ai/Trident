add_mlir_tool(trident-core-opt trident-core-opt.cc)

set_target_properties(trident-core-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${PROJECT_BINARY_DIR}/bin")

set(_trident_core_opt_cuda_libraries)
if(NOT TRIDENT_ASCEND_BACKEND)
  list(APPEND _trident_core_opt_cuda_libraries CUDA::cudart CUDA::cupti)
endif()
target_link_libraries(
  trident-core-opt
  PRIVATE ${_trident_core_opt_cuda_libraries} MLIRMlirOptMain
          MLIRRegisterAllPasses TVMFFIDialect TridentUtilsRegistration
          ${TORCH_LIBRARIES})

mlir_check_all_link_libraries(trident-core-opt)
