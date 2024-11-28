# Minimum required version of CMake
cmake_minimum_required(VERSION 3.12)

# Project name
project(Bang)

# Enable CUDA support (required if CUDA is present)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ../../utils
)

# Set compiler options for C++ and OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

# Set options for nvcc
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp -std=c++14 --compiler-options '-fPIC'")

# Define the shared library target (bang)
cuda_add_library(bang SHARED parANN.cu)

# Ensure position-independent code and separable compilation
set_target_properties(bang PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify the test driver executable
add_executable(bang_search test_driver.cpp)
target_link_libraries(bang_search PRIVATE bang dl)

# Clean rule (optional, CMake handles this with 'make clean')
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY CLEAN_DIRECTORIES "bang.so;bang_search")
