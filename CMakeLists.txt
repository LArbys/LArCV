cmake_minimum_required(VERSION 3.10)
project(larcv VERSION 1.1.0)

#---Install location by default is in the build directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "${CMAKE_BUILD_DIR}" FORCE )
endif()

#---Module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

## Dependencies

#---CERN ROOT
find_package(ROOT CONFIG REQUIRED)

#---OPENCV
if (USE_OPENCV)
  set(OpenCV_DIR $ENV{OPENCV_LIBDIR}/../share/OpenCV)
  find_package(OpenCV CONFIG REQUIRED)
  if (NOT OpenCV_FOUND)
    set(USE_OPENCV 0)
  endif()
endif()

#---PYTHON

if (USE_PYTHON3)
  find_package( PythonInterp 3 REQUIRED )
  find_package( PythonLibs 3 REQUIRED )
  # warning gross hack
  execute_process( COMMAND python3 -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE )
  #message(${NUMPY_INCLUDES})
  set(LARCV_USE_PYTHON 1)
  set(LARCV_PYTHON_VERSION 3)
elseif(USE_PYTHON2)
  find_package( PythonInterp 2.7 REQUIRED )
  find_package( PythonLibs 2.7 REQUIRED )
  find_package( Numpy REQUIRED )
  set(LARCV_USE_PYTHON 1)
  set(LARCV_PYTHON_VERSION 2)
else()
  set(LARCV_USE_PYTHON 0)
endif()

#---larbys GEO2D
# use it if env variables defined
if (DEFINED ENV{GEO2D_BASEDIR})
  set(LARCV_USE_GEO2D 1)
else()
  set(LARCV_USE_GEO2D 0)
endif()

# unless explicitly told no
if (NOT USE_GEO2D)
  set(LARCV_USE_GEO2D 0)
endif()

if (LARCV_USE_GEO2D)
  set( GEO2D_INC_DIR "$ENV{GEO2D_INCDIR}" )
  set( GEO2D_LIB_DIR "$ENV{GEO2D_LIBDIR}" )
  file( GLOB GEO2D_LIBS "$ENV{GEO2D_LIBDIR}/*.so" )
endif()

# --- Torch
if (USE_TORCH)
  if (ON_FNAL)
    # For FNAL
    set( Torch_DIR "$ENV{LIBTORCH_FQ_DIR}/lib/python2.7/site-packages/torch/share/cmake/Torch" )
    #set( OpenBLAS_DIR "$ENV{OPENBLAS_FQ_DIR}/lib/cmake/openblas" )
    #find_package( OpenBLAS REQUIRED )    
  else()
    set( Torch_DIR "$ENV{LIBTORCH_FQ_DIR}/lib/python2.7/site-packages/torch/share/cmake/Torch" )
    find_package( CUDA )
  endif()

  find_package( Torch REQUIRED )
endif()


## Git Submodules

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

#---check json
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/json/CMakeLists.txt")
  message(FATAL_ERROR "The json submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

## Setup

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

#---Define Variables for subdirectory cmakelists
set(LARCV_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

## External repos (from submodules)
include_directories(extern/json)
add_subdirectory(extern/json)
set(LARCV_JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/json/single_include)

## Add Source

#---Add base of source
add_subdirectory(larcv)

#---Configuration export
include(CMakePackageConfigHelpers)
set(LARCV_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(LARCV_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(TARGETS_EXPORT_NAME "larcvTargets")
configure_package_config_file("cmake/Config.cmake.in"
  "${CMAKE_BINARY_DIR}/cmake/larcvConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/larcv"
  PATH_VARS LARCV_INCLUDE_DIR LARCV_LIBRARY_DIR LARCV_BASE_DIR LARCV_USE_GEO2D LARCV_USE_PYTHON LARCV_PYTHON_VERSION)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/cmake/larcvConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion)


## Export Targets

#---Create info to include LArCV in other cmake files
install(EXPORT larcv
        FILE larcvTargets.cmake
        DESTINATION "${CMAKE_BINARY_DIR}/installed/lib/cmake/larcv")

#---Create info to include LArCV in other cmake files
install(FILES "${CMAKE_BINARY_DIR}/cmake/larcvConfig.cmake"
  "${CMAKE_BINARY_DIR}/cmake/larcvConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/larcv")
