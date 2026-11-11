if(NOT FLAGTREE_BACKEND STREQUAL "hcu")
  return()
endif()

set(_trident_hcu_prefixes)
foreach(_variable DTK_PATH ROCM_PATH HIP_PATH)
  if(NOT "$ENV{${_variable}}" STREQUAL "")
    list(APPEND _trident_hcu_prefixes "$ENV{${_variable}}"
         "$ENV{${_variable}}/hip")
  endif()
endforeach()
list(APPEND _trident_hcu_prefixes "/opt/dtk" "/opt/dtk/hip")

set(_trident_saved_cmake_prefix_path "${CMAKE_PREFIX_PATH}")
list(PREPEND CMAKE_PREFIX_PATH ${_trident_hcu_prefixes})
find_package(hip CONFIG REQUIRED)
set(CMAKE_PREFIX_PATH "${_trident_saved_cmake_prefix_path}")

add_library(TridentHcuRuntime SHARED HcuRuntime.cc)
target_link_libraries(TridentHcuRuntime PRIVATE hip::host)
set_target_properties(
  TridentHcuRuntime
  PROPERTIES INSTALL_RPATH "$ORIGIN"
             INSTALL_RPATH_USE_LINK_PATH ON)

install(
  TARGETS TridentHcuRuntime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
