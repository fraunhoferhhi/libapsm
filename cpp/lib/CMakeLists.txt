
#https://devblogs.nvidia.com/building-cuda-applications-cmake/

# find all C++ CUDA source files in this folder
#file( GLOB_RECURSE SOURCES_APSM_LIBRARY
#    *.cu
#    *.cuh
#)
aux_source_directory( apsm SOURCES_APSM_LIBRARY )
aux_source_directory( noma SOURCES_NOMA_LIBRARY )
aux_source_directory( cuda SOURCES_CUDA_LIBRARY )

### Make the library ######################################################

message( STATUS "Building APSM library" )
# create an library in which the tests will be stored
add_library( apsm ${SOURCES_APSM_LIBRARY} ${SOURCES_CUDA_LIBRARY} ${SOURCES_NOMA_LIBRARY} )

# Create a static library to link the code to the tests
target_include_directories( apsm PUBLIC ${CMAKE_SOURCE_DIR}/lib ${CUDA_INCLUDE_DIRS} ${THRUST_INCLUDE_DIRS} )

# we using separate cu files in lib
set_target_properties( apsm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# cublas is used from noma lls
target_link_libraries( apsm json cublas )

# compiler more strict
target_compile_options( apsm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings,cross-execution-space-call -Xcompiler -fPIC> )

# copy or link the data directory into build folder
add_custom_command(
  TARGET apsm
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink  # copy_directory or create_symlink
    ${CMAKE_SOURCE_DIR}/data
    ${CMAKE_BINARY_DIR}/dataset
)
