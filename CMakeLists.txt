cmake_minimum_required (VERSION 3.0)
project( "SimpleNetwork" )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_property( GLOBAL PROPERTY USE_FOLDERS ON )

###############################################################
# SimpleNetwork Library                                       #
###############################################################

# Common files
file( GLOB LIBRARY_COMMON_HEADER_FILES "common/*.h" )
file( GLOB LIBRARY_COMMON_SOURCE_FILES "common/*.cpp" )
source_group( "Common\\Header Files" FILES ${LIBRARY_COMMON_HEADER_FILES} )
source_group( "Common\\Source File" FILES ${LIBRARY_COMMON_SOURCE_FILES} )

# SimpleNetwork files
file( GLOB NETWORK_HEADER_FILES "src/*.h" )
file( GLOB NETWORK_SOURCE_FILES "src/*.cpp" )
source_group( "SimpleNetwork\\Header Files" FILES ${NETWORK_HEADER_FILES} )
source_group( "SimpleNetwork\\Source File" FILES ${NETWORK_SOURCE_FILES} )

# Recipes Library generation
add_library( SimpleNetwork ${LIBRARY_COMMON_HEADER_FILES} ${LIBRARY_COMMON_SOURCE_FILES} ${NETWORK_HEADER_FILES} ${NETWORK_SOURCE_FILES} )
target_include_directories( SimpleNetwork PUBLIC "common" "src" )

###############################################################
# Sample Project                                              #
###############################################################

	file( GLOB SAMPLES RELATIVE "${CMAKE_CURRENT_LIST_DIR}/samples" "${CMAKE_CURRENT_LIST_DIR}/samples/*" )
	FOREACH( SAMPLE ${SAMPLES} )
			file( GLOB SAMPLE_HEADER_FILES "samples/${SAMPLE}/*.h" )
			file( GLOB SAMPLE_SOURCE_FILES "samples/${SAMPLE}/*.cpp")
			source_group( "Source File" FILES ${SAMPLE_SOURCE_FILES} )
			source_group( "Header File" FILES ${SAMPLE_HEADER_FILES} )
			add_executable( ${SAMPLE} ${SAMPLE_HEADER_FILES} ${SAMPLE_SOURCE_FILES} )
			target_link_libraries( ${SAMPLE} SimpleNetwork)
			target_include_directories( ${SAMPLE} PUBLIC "src" "common" )
	ENDFOREACH()