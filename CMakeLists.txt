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
  if (NOT OpenCV_FOUND)
    set(USE_OPENCV 0)
  endif()
endif()

#---PYTHON
if (USE_PYTHON3)
  find_package( PythonInterp 3 REQUIRED )
  find_package( PythonLibs 3 REQUIRED )
  set(LARCV_USE_PYTHON 1)
elseif(USE_PYTHON2)
  find_package( PythonInterp 2.7 REQUIRED )
  find_package( PythonLibs 2.7 REQUIRED )
  set(LARCV_USE_PYTHON 1)
else()
  set(LARCV_USE_PTYHON 0)
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






