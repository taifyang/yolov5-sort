project(SORT)
cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCV2_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/source})
aux_source_directory(${PROJECT_SOURCE_DIR}/source SRCS)
file(GLOB INC_PATH ${PROJECT_SOURCE_DIR}/include/*.h)
list(APPEND INCS ${INC_PATH} )

add_executable(sort main.cpp ${INCS} ${SRCS} )
target_link_libraries(sort ${OpenCV_LIBRARIES})