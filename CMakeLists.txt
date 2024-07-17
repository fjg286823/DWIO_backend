cmake_minimum_required(VERSION 3.17)
project(DWIO_hash)

set(CMAKE_CXX_STANDARD 14)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# ------------------------------------------------
#                 Dependencies
# ------------------------------------------------

## Pangolin
find_package(Pangolin REQUIRED)

## Eigen3
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    message(${EIGEN3_INCLUDE_DIR})
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)

## PCL
find_package(PCL REQUIRED QUIET)
if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
ENDIF ()

## CUDA
find_package(CUDA 11.0 REQUIRED)
IF (CUDA_FOUND)
    message(${CUDA_INCLUDE_DIRS})
    include_directories("${CUDA_INCLUDE_DIRS}")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14 -gencode arch=compute_86,code=sm_86)
ENDIF ()

## OpenCV
find_package(OpenCV 4.5.0  REQUIRED)
if (OpenCV_INCLUDE_DIRS)
    message(${OpenCV_INCLUDE_DIRS})
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories(${OpenCV_DIR}/lib)
endif (OpenCV_INCLUDE_DIRS)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

add_executable(DWIO_hash ${PROJECT_SOURCE_DIR}/main.cpp)


include_directories("${PROJECT_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/cuda/include" "${PROJECT_SOURCE_DIR}/hash")
file(GLOB DWIO_SRCS "${PROJECT_SOURCE_DIR}/cuda/*.cu" "${PROJECT_SOURCE_DIR}/*.cpp")

cuda_add_library(DWIO_lib STATIC "${DWIO_SRCS}")

target_link_libraries(DWIO_hash "${OpenCV_LIBS}" DWIO_lib)
target_link_libraries(DWIO_hash ${Pangolin_LIBRARIES} ${PCL_LIBRARIES})
