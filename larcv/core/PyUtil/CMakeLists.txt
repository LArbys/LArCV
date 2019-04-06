set(MODULE_NAME LArCVCorePyUtil)

# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")

# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

# includes
include_directories( ${LARCV_BASE_DIR} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES})
#message("NUMPY_INCLUDES: ${NUMPY_INCLUDES}")

# compiler flag for python3
#message( "PYTHON VERSION: ${LARCV_PYTHON_VERSION}" )
if (${LARCV_PYTHON_VERSION} EQUAL  3)
  add_definitions(-DUSE_PYTHON3)
endif()

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)

# Generate the shared library from the sources
add_library( ${MODULE_NAME} SHARED ${SOURCES} G__${MODULE_NAME}.cxx)
set_target_properties( ${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries( ${MODULE_NAME} ${PYTHON_LIBRARIES} LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor )

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/PyUtil)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )
