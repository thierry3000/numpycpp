cmake_minimum_required(VERSION 2.8.8)
project(numpycpp)

# set a search path for additional custom cmake modules.
# according to recommendation in https://cmake.org/Wiki/CMake:How_To_Find_Libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/.cmake/modules/")

set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost REQUIRED COMPONENTS "python")
find_package(NumPy REQUIRED)

#from cmake 2.8 module documentation:
#The Python_ADDITIONAL_VERSIONS variable can be used to specify a list of version numbers that should be # taken into account when searching for Python. You need to set this variable before calling find_package(PythonLibs).

#for manual configuration, set PYTHON_INCLUDE_DIR and PYTHON_LIBRARY
#see also do-configure-tumorcode-lusi.sh script

# add_custom_target(docs ALL
#                   COMMAND naturaldocs -i . -p doc -xi doc -xi htmldoc -o HTML htmldoc
#                   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#                   SOURCES numpy.hpp)

message(STATUS "PYTHON include dirs = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON libs         = ${PYTHON_LIBRARIES}")
message(STATUS "NumPy include dirs  = ${PYTHON_NUMPY_INCLUDE_DIR}")

include_directories(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
include_directories(.)

add_library(demo SHARED
            numpy.cpp demo.cpp)
target_link_libraries(demo ${Boost_PYTHON_LIBRARY_RELEASE} ${PYTHON_LIBRARIES})

# install(DIRECTORY ${CMAKE_SOURCE_DIR}/py
# 	DESTINATION .
# 	FILES_MATCHING PATTERN "*.py")
