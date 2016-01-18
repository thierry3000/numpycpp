cmake_minimum_required(VERSION 2.8.8)
project(numpycpp)

# This builds the demo only. You probably want to just incorporate numpy.hpp and numpy.cpp into your project. 
# It is possible to use (sym) links for that but beware that stupid editors like kdevelop might break the
# link by deleting the link and then rewriting a new copy of the file to disk ...

# set a search path for additional custom cmake modules.
# according to recommendation in https://cmake.org/Wiki/CMake:How_To_Find_Libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/.cmake/modules/")

set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost REQUIRED COMPONENTS "python")
find_package(NumPy REQUIRED)

message(STATUS "PYTHON include dirs = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON libs         = ${PYTHON_LIBRARIES}")
message(STATUS "NumPy include dirs  = ${PYTHON_NUMPY_INCLUDE_DIR}")

include_directories(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
include_directories(.)

add_library(demo SHARED
            numpy.cpp demo.cpp)

add_dependencies(demo demo.py)

target_link_libraries(demo ${Boost_PYTHON_LIBRARY_RELEASE} ${PYTHON_LIBRARIES})

add_custom_command(TARGET demo POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/demo.py ${CMAKE_BINARY_DIR}/demo.py)