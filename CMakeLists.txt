cmake_minimum_required(VERSION 3.18)
project(dngscan LANGUAGES CXX)

# Native backend defaults ON (review batch 17): a standard install gets the
# C++ AgX/HDR/output kernels; a toolchain that cannot compile them falls
# back to the pure-Python wheel with a warning instead of failing the build.
option(DNGSCAN_BUILD_NATIVE "Build the optional _dngscan_fast extension" ON)

if(DNGSCAN_BUILD_NATIVE)
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles("int main() { return 0; }" DNGSCAN_CXX_WORKS)
  if(DNGSCAN_CXX_WORKS)
    add_subdirectory(cpp)
  else()
    message(WARNING
      "dngscan: no working C++ toolchain — building the pure-Python wheel; "
      "the NumPy reference path stays fully functional")
  endif()
endif()
