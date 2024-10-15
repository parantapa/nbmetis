cmake_minimum_required(VERSION 3.27)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Debug or Release")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif(CCACHE_FOUND)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

# Setup for Nanobind

find_package(Python 3.11
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

add_subdirectory(extern/nanobind)

add_compile_definitions(
"REALTYPEWIDTH=32"
"IDXTYPEWIDTH=32"
)

add_subdirectory(extern/GKlib)
add_subdirectory(extern/METIS)

set(METIS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/METIS/include")

# nbmetis proper

nanobind_add_module(_nbmetis STABLE_ABI src/nbmetis.cpp)
target_link_libraries(_nbmetis PRIVATE metis)
target_include_directories(_nbmetis PRIVATE "${METIS_INCLUDE_DIR}")

nanobind_add_stub(
  _nbmetis_stub
  MODULE _nbmetis
  OUTPUT _nbmetis.pyi
  DEPENDS _nbmetis
)

install(TARGETS _nbmetis LIBRARY DESTINATION nbmetis)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_nbmetis.pyi DESTINATION nbmetis)
