set(
    ipddp_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_base.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_core.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_ocp.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_dynamics.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_utils.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_backwardpass.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ipddp_forwardpass.hpp"
    )

set(ipddp_headers "${ipddp_headers}" PARENT_SCOPE)


