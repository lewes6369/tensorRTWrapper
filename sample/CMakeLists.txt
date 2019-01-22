cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

#add lib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../code sample.out)
include_directories({$CMAKE_CURRENT_SOURCE_DIR}/../code)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

#build classifer
add_executable(classifer evalClassifNet.cpp eval.cpp dataReader.cpp)
target_link_libraries(classifer TrtNet ${OpenCV_LIBS})

#build runNet
add_executable(runNet runNet.cpp)
target_link_libraries(runNet TrtNet ${OpenCV_LIBS})

#build runTwoNets
add_executable(runTwoNets runTwoNets.cpp)
target_link_libraries(runTwoNets TrtNet ${OpenCV_LIBS})

install(TARGETS classifer DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
install(TARGETS runNet DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)
install(TARGETS runTwoNets DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)