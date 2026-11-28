cmake_minimum_required(VERSION 3.24)
project(spiral LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_path(LZ4_INCLUDE_DIR lz4.h HINTS /opt/homebrew/opt/lz4/include REQUIRED)
find_library(LZ4_LIBRARY lz4 HINTS /opt/homebrew/opt/lz4/lib REQUIRED)

add_executable(spiral_fit src/main.cpp src/io.cpp src/model.cpp src/tracks.cpp src/losses.cpp src/export.cpp src/gpu.mm
    src/crossings.cpp src/patches.cpp)
target_compile_options(spiral_fit PRIVATE -O3 -mcpu=apple-m1 -ffp-contract=off)
set_source_files_properties(src/gpu.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
target_include_directories(spiral_fit PRIVATE ${LZ4_INCLUDE_DIR})
target_link_libraries(spiral_fit PRIVATE ${LZ4_LIBRARY} "-framework Metal" "-framework Foundation")
