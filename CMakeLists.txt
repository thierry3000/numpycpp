cmake_minimum_required(VERSION 2.8.8)
project(numpycpp)

# set a search path for additional custom cmake modules.
# according to recommendation in https://cmake.org/Wiki/CMake:How_To_Find_Libraries
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/.cmake/modules/")

set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost REQUIRED COMPONENTS "python")

message("found Boost_VERSION: ${Boost_VERSION}")
#since boost 1.63
if(Boost_VERSION GREATER 106500)
  message(STATUS "New boost_numpy detected, no need to use this piece of code!")
  find_package(Boost REQUIRED COMPONENTS "numpy")
endif()
find_package(NumPy REQUIRED)

message(STATUS "PYTHON include dirs = ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON libs         = ${PYTHON_LIBRARIES}")
message(STATUS "NumPy include dirs  = ${PYTHON_NUMPY_INCLUDE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

include_directories(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
include_directories(.)

if(Boost_VERSION LESS 106500)
  add_library(numpycpp STATIC numpy.cpp)
  add_library(demo SHARED demo.cpp)
else()
  add_library(demo SHARED demo_boost_numpy.cpp)
endif()



#message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
#add_dependencies(demo "${CMAKE_SOURCE_DIR}/demo.py")
message(STATUS "found Boost_PYTHON_LIBRARY:      ${Boost_PYTHON_LIBRARY}")
message(STATUS "found Boost_LIBRARIES:       ${Boost_LIBRARIES}")

if(Boost_VERSION GREATER 106500)
  target_link_libraries(demo boost_python boost_numpy )
else()
  target_link_libraries(demo numpycpp ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} )
endif()
add_custom_command(TARGET demo POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/demo.py ${CMAKE_BINARY_DIR}/demo.py)

if(Boost_VERSION LESS 106500)
  install(TARGETS numpycpp ARCHIVE DESTINATION lib)
  install(FILES numpy.hpp DESTINATION include)
endif()
