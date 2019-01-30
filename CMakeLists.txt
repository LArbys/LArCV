cmake_minimum_required(VERSION 3.10)
project(larcv)

#---Install location by default is in the build directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "${CMAKE_BUILD_DIR}" FORCE )
endif()

#---ROOT
find_package(ROOT REQUIRED)

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

#---Add base of source
add_subdirectory(larcv)

#---Create info to include LArCV in other cmake files
install(EXPORT larcv
        FILE larcvTargets.cmake
        DESTINATION "${CMAKE_BINARY_DIR}/installed/lib/cmake/larcv")






