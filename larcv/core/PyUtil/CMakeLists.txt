set(MODULE_NAME LArCVCorePyUtil)

# Collect the headers
set( HEADERS 
  load_pyutil.h
  PyUtils.h
  NumpyArray.h
  PyImageMaker.h
  PyImageStitcher.h
)

# Collect the sources
add_library( ${MODULE_NAME} SHARED 
  load_pyutil.cxx
  PyUtils.cxx
  NumpyArray.cxx
  PyImageMaker.cxx
  PyImageStitcher.cxx
)

set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_include_directories(${MODULE_NAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larcv/core/PyUtil>
  PRIVATE
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries( ${MODULE_NAME} PUBLIC LArCVCoreBase LArCVCoreDataFormat )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR} 
  ${PYTHON_INCLUDE_PATH} 
  ${NUMPY_INCLUDES} 
)

# compiler flag for python3
if (${LARCV_PYTHON_VERSION} EQUAL  3)
  #message( STATUS "PYTHON VERSION: ${LARCV_PYTHON_VERSION}" )
  add_definitions(-DUSE_PYTHON3)
endif()

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)

# add dictionary file to sources
target_sources( ${MODULE_NAME} PRIVATE G__${MODULE_NAME}.cxx )

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/PyUtil)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )
