cmake_minimum_required(VERSION 3.18)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

pybind11_add_module(
  _dngscan_fast
  src/bindings.cpp
  src/agx_core.cpp
  src/hdr_core.cpp
  src/output_core.cpp
)

target_include_directories(_dngscan_fast PRIVATE include)
target_compile_features(_dngscan_fast PRIVATE cxx_std_17)
target_link_libraries(_dngscan_fast PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(_dngscan_fast PRIVATE /O2 /fp:precise)
else()
  target_compile_options(_dngscan_fast PRIVATE -O3 -ffp-contract=off)
endif()

install(TARGETS _dngscan_fast DESTINATION dngscan)
