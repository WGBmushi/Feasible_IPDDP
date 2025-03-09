cmake_minimum_required(VERSION 3.10)
project(ipddp VERSION 0.1.0 LANGUAGES CXX)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${MATPLOTLIB_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")  

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(SymEngine REQUIRED)
include_directories(${SYMENGINE_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(examples)

# IPDDP
add_library(ipddp SHARED ${ipddp_src} ${ipddp_headers})
target_link_libraries(ipddp ${SYMENGINE_LIBRARIES} ${PYTHON_LIBRARIES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)




