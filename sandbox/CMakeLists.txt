cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

set(CUDA_NVCC_FLAGS -arch=compute_75 -code=sm_75)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(main main.cpp)

target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})