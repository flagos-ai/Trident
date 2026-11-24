add_mlir_public_c_api_library(
  TridentCoreCAPI
  Dialects.cc
  Registration.cc
  TVMFFIABITypes.cc
  TVMFFIAPITypes.cc
  TVMFFIMonadTypes.cc
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ENABLE_AGGREGATION
  LINK_LIBS
  PUBLIC
  DLPackDialect
  MLIRIR
  MLIRSupport
  TVMFFIABIDialect
  TVMFFIAPIDialect
  TVMFFIArithDialect
  TVMFFIMonadDialect
  TridentUtilsRegistration)

install(
  TARGETS TridentCoreCAPI
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
