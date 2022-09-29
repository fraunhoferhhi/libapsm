# ##############################################################################
# Add libraries
# ##############################################################################

# Note : add_library( name INTERFACE ) can be used to add header-only libraries

# binary
add_library( binary INTERFACE )
target_include_directories( binary INTERFACE binary/include )

# argparse
add_library( argparse INTERFACE )
target_include_directories( argparse INTERFACE argparse/include )

# json
add_library( json INTERFACE )
target_include_directories( json INTERFACE json/include )
