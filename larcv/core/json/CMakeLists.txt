set(MODULE_NAME LArCVCoreJSON)

# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")
 
# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

# Include directory
include_directories( ${LARCV_JSON_INCLUDE_DIR} ${LARCV_BASE_DIR})

if (LARCV_USE_PYTHON)
  add_definitions("-DHASPYUTIL")
  include_directories( ${PYTHON_INCLUDE_PATH} )
endif()

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)
 
# Generate the shared library from the sources
add_library(${MODULE_NAME} SHARED ${SOURCES} G__${MODULE_NAME}.cxx)
set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries(${MODULE_NAME} PUBLIC ${ROOT_LIBS} LArCVCoreDataFormat )

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/json)
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )


