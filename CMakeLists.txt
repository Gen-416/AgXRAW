cmake_minimum_required(VERSION 3.18)
project(dngscan LANGUAGES CXX)

option(DNGSCAN_BUILD_NATIVE "Build the optional _dngscan_fast extension" OFF)

if(DNGSCAN_BUILD_NATIVE)
  add_subdirectory(cpp)
endif()
