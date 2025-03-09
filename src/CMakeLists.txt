set(
    ipddp_src
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_dynamics.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_backwardpass.cpp"    
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_forwardpass.cpp"
)
set(ipddp_src "${ipddp_src}" PARENT_SCOPE)






