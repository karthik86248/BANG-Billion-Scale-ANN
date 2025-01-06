# Minimum required version of CMake
cmake_minimum_required(VERSION 3.24)

set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Project name
project(bang)

# Enable CUDA support (required if CUDA is present)
find_package(CUDA REQUIRED)

include(cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(bang)

if(NOT TARGET raft::raft)
   find_package(raft)
 endif()

# Include directories
include_directories(
    ../../utils
)

# Set compiler options for C++ and OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

# Set options for nvcc
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")

# Define the shared library target (bang)
cuda_add_library(bang SHARED parANN.cu)

# Ensure position-independent code and separable compilation
set_target_properties(bang PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(bang raft::raft)

# Specify the test driver executable
# add_executable(bang_search test_driver.cpp)
# target_link_libraries(bang_search bang dl)

# Clean rule (optional, CMake handles this with 'make clean')
# set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY CLEAN_DIRECTORIES "bang.so;bang_search")
