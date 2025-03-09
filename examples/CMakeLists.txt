cmake_minimum_required(VERSION 3.10)

# --------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(ocp_inverted_pendulum ocp_inverted_pendulum.cpp)
target_link_libraries(ocp_inverted_pendulum PRIVATE ipddp)  



