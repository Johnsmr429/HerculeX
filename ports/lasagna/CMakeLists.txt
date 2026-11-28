cmake_minimum_required(VERSION 3.24)
project(lasagna LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_path(LZ4_INCLUDE_DIR lz4.h HINTS /opt/homebrew/opt/lz4/include REQUIRED)
find_library(LZ4_LIBRARY lz4 HINTS /opt/homebrew/opt/lz4/lib REQUIRED)
find_path(IGL_INCLUDE_DIR igl/signed_distance.h HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/deps/libigl/libigl-2.6.0/include REQUIRED)
find_path(EIGEN_INCLUDE_DIR Eigen/Dense HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime/deps/eigen/eigen-3.4.0 /opt/homebrew/include/eigen3 REQUIRED)

add_executable(lasagna_fit src/main.cpp src/io.cpp src/data.cpp src/shell.cpp src/cylinder.cpp src/pyramid.cpp
    src/barrier.cpp src/fit.cpp src/mesh.cpp)
target_compile_options(lasagna_fit PRIVATE -O3 -mcpu=apple-m1 -ffp-contract=off)
set_source_files_properties(src/barrier.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=on)
target_include_directories(lasagna_fit PRIVATE ${LZ4_INCLUDE_DIR} ${IGL_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR})
target_link_libraries(lasagna_fit PRIVATE ${LZ4_LIBRARY})
