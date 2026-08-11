cmake_minimum_required(VERSION 3.18)
# A9 item 7: LANGUAGES NONE — initialising the project with CXX made the
# advertised pure-Python fallback unreachable (configure itself failed on
# hosts without a C++ toolchain, before the option below was ever read).
# CXX is enabled OPTIONALLY after probing for a compiler.
project(dngscan LANGUAGES NONE)

# Native backend defaults ON (review batch 17): a standard install gets the
# C++ AgX/HDR/output kernels; a toolchain that cannot compile them falls
# back to the pure-Python wheel with a warning instead of failing the build.
option(DNGSCAN_BUILD_NATIVE "Build the optional _dngscan_fast extension" ON)

if(DNGSCAN_BUILD_NATIVE)
  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    add_subdirectory(cpp)
  else()
    message(WARNING
      "dngscan: no working C++ toolchain — building the pure-Python wheel; "
      "the NumPy reference path stays fully functional")
  endif()
endif()
