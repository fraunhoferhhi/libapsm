cmake_minimum_required( VERSION 3.21 FATAL_ERROR )

##################################################################
# Set new Policies of cmake 3.22
##################################################################

# https://www.mankier.com/7/cmake-policies#

# cmake minimum required implicitly call cmake_policy( VERSION )
#cmake_policy( VERSION 3.21.0 )

if( POLICY CMP0127 )
    # policies not known to CMake 3.21 ...
    #cmake_policy( SET CMP0127 NEW )
    set( CMAKE_POLICY_DEFAULT_CMP0127 NEW )
endif( POLICY CMP0127 )

if( POLICY CMP0128 )
    # policies not known to CMake 3.21 ...
    #cmake_policy( SET CMP0128 NEW )
    set( CMAKE_POLICY_DEFAULT_CMP0128 NEW )
endif( POLICY CMP0128 )

##################################################################
# Settings
##################################################################

project ( NOMA
    VERSION 0.5.0
    LANGUAGES C CXX CUDA
)

#set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ) # static libraries like `.a` or `.lib`
#set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ) # shared libaries like `.so`
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin ) # binaries created by cmake

set( CMAKE_INSTALL_PREFIX install )

##################################################################
# Set a default build type to release if none was specified
##################################################################

set( default_build_type "Release" )

if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
  message( STATUS "Setting build type to '${default_build_type}' as none was specified." )
  set( CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE )
  # Set the possible values of build type for cmake-gui
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )
endif()

##################################################################
# GIT commit hash
##################################################################

set( git_cmd "git" )
set( git_arg1 "log" )
set( git_arg2 "-1" )
set( git_arg3 "--format=%h" )
execute_process( COMMAND ${git_cmd} ${git_arg1} ${git_arg2} ${git_arg3}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE GIT_RESULT
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if( GIT_RESULT EQUAL "0" )
    message( STATUS "APSM git commit hash[ ${GIT_COMMIT_HASH} ]" )
    add_definitions( "-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"" )
endif()

##################################################################
# Dependencies
##################################################################

set( CUDA_MIN_VERSION 10.0 )

# Add CUDA and Thrust dependencies to this project
find_package( CUDAToolkit ${CUDA_MIN_VERSION} REQUIRED )

##################################################################
# Configure Compiler CXX and CUDA
##################################################################

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set( CMAKE_CXX_STANDARD 17 ) # 17 or 20
set( CMAKE_CXX_STANDARD_REQUIRED TRUE )
set( CMAKE_CXX_EXTENSIONS OFF )

set( CMAKE_CUDA_STANDARD 17 ) # 17 or 20
set( CMAKE_CUDA_STANDARD_REQUIRED TRUE )
set( CMAKE_CUDA_EXTENSIONS OFF )
set( CMAKE_CUDA_ARCHITECTURES OFF )

# Warning options can't be passed directly to nvcc, this solution was inspired by https://stackoverflow.com/a/59426558
#list( APPEND cxx_warning_flags "-Wall" "-Wextra" "-Werror" )
foreach( flag IN LISTS cxx_warning_flags )
    add_compile_options( "$<$<COMPILE_LANGUAGE:CXX>:${flag}>" )
    add_compile_options( "$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=${flag}>" )
endforeach()

# setting to prevent errors during compiling
list( APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c" )
list( APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion" )
list( APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing" )
list( APPEND CUDA_NVCC_FLAGS "--extended-lambda" )
list( APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr" )
list( APPEND CUDA_NVCC_FLAGS "-fPIC" )

find_program( CCACHE_PROGRAM ccache )
if( USE_CCACHE AND CCACHE_PROGRAM )
    message ( STATUS "Using ccache to speed up compilation" )
    set_property( GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}" )
ENDIF( USE_CCACHE AND CCACHE_PROGRAM )

##################################################################
# Configure Debug
##################################################################

if( CMAKE_BUILD_TYPE MATCHES Debug )
    # Host debug (--debug/-g)
    if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-g" )
        list ( APPEND CUDA_NVCC_FLAGS "-g" )
    endif ()

    # Device debug
    if ( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-G" )
        list ( APPEND CUDA_NVCC_FLAGS "-G" )
    endif ()
endif( CMAKE_BUILD_TYPE MATCHES Debug )

##################################################################
# Setup CUDA architecture
##################################################################

# https://devblogs.nvidia.com/building-cuda-applications-cmake/
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

if( CMAKE_CUDA_COMPILER_LOADED ) # CUDA as a language
  if( CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)" )
    set( CUDA_VERSION "${CMAKE_MATCH_1}" )
  endif()
endif()

# start with an empty list
set( GPU_COMPUTE_CAPABILITIES "" )

# add Maxwell cards (CUDA 6 until CUDA 11)
# SM50 - Tesla/Quadro M series
# SM52 - Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X
# SM53 - Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano
if( CUDA_VERSION VERSION_GREATER_EQUAL "6.0" AND CUDA_VERSION VERSION_LESS "12.0" )
    message("CUDA  6.0 and later detected, adding compute capability for Maxwell cards.")
    list( APPEND GPU_COMPUTE_CAPABILITIES 52 )
endif()

# add Pascal cards (CUDA 8 and later)
# SM60 - Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
# SM61 - GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030 (GP108), GT 1010 (GP108) Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
# SM62 - Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2
if( CUDA_VERSION VERSION_GREATER_EQUAL "8.0" )
    message("CUDA  8.0 and later detected, adding compute capability for Pascal cards.")
    list( APPEND GPU_COMPUTE_CAPABILITIES 60 61 )
endif()

# add Volta cards (CUDA 9 and later)
# SM70 - DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
# SM72 - Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX
if( CUDA_VERSION VERSION_GREATER_EQUAL "9.0" )
    message("CUDA  9.0 and later detected, adding compute capability for Volta cards.")
    list( APPEND GPU_COMPUTE_CAPABILITIES 70 )
endif()

# add Turing cards (CUDA 10 and later)
# SM75 - GTX/RTX Turing – GTX 1660 Ti, RTX 2060, RTX 2070, RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4
if( CUDA_VERSION VERSION_GREATER_EQUAL "10.0" )
    message("CUDA 10.0 and later detectet, adding compute capability for Turing cards.")
    list( APPEND GPU_COMPUTE_CAPABILITIES 75 )
endif()

# add Ampere cards (CUDA 11.1 and later)
# SM80 - NVIDIA A100 (the name “Tesla” has been dropped – GA100), NVIDIA DGX-A100
# SM86 - Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A2000, A3000, RTX A4000, A5000, A6000, NVIDIA A40, GA106 – RTX 3060, GA104 – RTX 3070, GA107 – RTX 3050, RTX A10, RTX A16, RTX A40, A2 Tensor Core GPU
# SM87 - for Jetson AGX Orin and Drive AGX Orin only
if( CUDA_VERSION VERSION_GREATER_EQUAL "11.1" )
    message("CUDA 11.1 and later detected, adding compute capability for Ampere cards.")
    list( APPEND GPU_COMPUTE_CAPABILITIES 80 86 87 )
endif()

# add Lovelace cards (CUDA 11.8 and later)
# SM89 – NVIDIA GeForce RTX 4090, RTX 4080, RTX 6000, Tesla L40
if( CUDA_VERSION VERSION_GREATER_EQUAL "11.8" )
    message("CUDA 11.8 and later detected, adding compute capability for Lovelace cards.")
    list( APPEND GPU_COMPUTE_CAPABILITIES 89 )
endif()

# add Hopper card (CUDA 12 and later)
# SM90 or SM_90, compute_90 – NVIDIA H100 (GH100)
if( CUDA_VERSION VERSION_GREATER_EQUAL "12.0" )
    message("CUDA 12.0 andl detected, adding compute capability for Hopper cards.")
    list( APPEND GPU_COMPUTE_CAPABILITIES 90 )
endif()

list( SORT GPU_COMPUTE_CAPABILITIES COMPARE NATURAL ORDER ASCENDING )

# create compiler arguments
list( GET GPU_COMPUTE_CAPABILITIES  0 MIN_GPU_ARCH )
list( GET GPU_COMPUTE_CAPABILITIES -1 MAX_GPU_ARCH )

if( NOT "${CMAKE_CUDA_FLAGS}" MATCHES "-gencode" )
    list ( APPEND CUDA_NVCC_FLAGS "-arch=sm_${MIN_GPU_ARCH}" )
    # GPU Compute Capability: real architecture (code=sm_X)
    foreach( GPU_CAP IN LISTS GPU_COMPUTE_CAPABILITIES )
        list ( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${GPU_CAP},code=sm_${GPU_CAP}" )
    endforeach()
    # GPU Compute Capability: virtual architecture (code=compute_X)
    list ( APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${MAX_GPU_ARCH},code=compute_${MAX_GPU_ARCH}" )
endif()

# add param list to CMAKE CUDA FLAGS
message( STATUS "CMAKE CUDA flags for nvcc: ${CMAKE_CUDA_FLAGS}" )
string( REPLACE ";" " " CUDA_NVCC_FLAGS_WITH_SPACE "${CUDA_NVCC_FLAGS}")
string( CONCAT CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " " ${CUDA_NVCC_FLAGS_WITH_SPACE} )
message( STATUS "CMAKE CUDA flags for nvcc: ${CMAKE_CUDA_FLAGS}" )

##################################################################
# Subprojects
##################################################################

# add libraries from external folder
# * argparse
# * binary
add_subdirectory( external )
add_subdirectory( lib )
add_subdirectory( cli )
