cmake_minimum_required(VERSION 3.1)
project(lane)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/lane.cpp)
set(SRC_MAIN ${CMAKE_SOURCE_DIR}/main.cpp)
set(NCNN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/ncnn)
set(SRC_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(NCNN_LIB ${CMAKE_SOURCE_DIR}/lib/libncnn.a)
# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -O0 -std=c++11 ")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g")
include_directories(${NCNN_INCLUDE_DIR} ${SRC_INCLUDE_DIR})

add_executable(lane ${SRC_DIR} ${OpenCV_INCLUDE_DIRS} ${SRC_MAIN})
target_link_libraries(lane ${OpenCV_LIBRARIES} ${NCNN_LIB})