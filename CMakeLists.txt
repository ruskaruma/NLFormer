cmake_minimum_required(VERSION 3.10)
project(NLFormer VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find required packages
find_package(nlohmann_json REQUIRED)
find_package(GTest REQUIRED)

# Include directories
include_directories(include)

# Create the main library
add_library(nlformer
    src/attention.cpp
    src/engine.cpp
    src/matcher.cpp
    src/types.cpp
)

# Link libraries
target_link_libraries(nlformer nlohmann_json::nlohmann_json)

# Create a demo executable
add_executable(demo
    examples/demo.cpp
)
target_link_libraries(demo nlformer)

# Create benchmark executable
add_executable(benchmark
    benchmarks/benchmark.cpp
)
target_link_libraries(benchmark nlformer)

# Create Python bindings (if pybind11 is available)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    add_subdirectory(python)
endif()

# Testing
enable_testing()
add_subdirectory(tests)

# Install targets
install(TARGETS nlformer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# CPack configuration for packaging
include(CPack)
