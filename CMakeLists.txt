cmake_minimum_required(VERSION 3.10)
project(larcv)

#---Install location by default is in the build directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "${CMAKE_BUILD_DIR}" FORCE )
endif()

## Dependencies

#---CERN ROOT
find_package(ROOT REQUIRED)

#---OPENCV
if (USE_OPENCV)
  find_package(OpenCV)
  message("Using OpenCV: ${OpenCV_FOUND}")
  if (NOT OpenCV_FOUND)
    set(USE_OPENCV 0)
  endif()
endif()

## Setup

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

#---Define Variables for subdirectory cmakelists
set(LARCV_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

## Add Source

#---Add base of source
add_subdirectory(larcv)

## Export Targets

#---Create info to include LArCV in other cmake files
install(EXPORT larcv
        FILE larcvTargets.cmake
        DESTINATION "${CMAKE_BINARY_DIR}/installed/lib/cmake/larcv")






