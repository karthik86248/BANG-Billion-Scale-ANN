set(RAPIDS_VERSION "24.12")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/BANG_RAPIDS.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${RAPIDS_VERSION}/RAPIDS.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/BANG_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/BANG_RAPIDS.cmake)