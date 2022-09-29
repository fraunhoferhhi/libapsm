file( GLOB COMMON_SRC_LIST
    common/argparse_helper.hpp
    common/argparse_helper.cpp
    common/binary_load.hpp
    common/binary_load.cpp
    common/util.hpp
    common/util.cpp
    common/modulation.hpp
    common/modulation.cpp
)

# use NOMA and CUDA lib
include_directories(
    ${CMAKE_SOURCE_DIR}/cli
    ${CMAKE_SOURCE_DIR}/lib
)

set( COMMON_LIBS binary argparse json )

### Make the executables ######################################################

message( STATUS "Building NOMA detect" )
aux_source_directory( detect DETECT_SRC_LIST )
add_executable( ${PROJECT_NAME}_detect ${COMMON_SRC_LIST} ${DETECT_SRC_LIST} )
target_include_directories( ${PROJECT_NAME}_detect PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
target_link_libraries( ${PROJECT_NAME}_detect apsm ${COMMON_LIBS} )
