set(MODULE_NAME LArCVCoreBase)

# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")
 
# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(${MODULE_NAME}Dict ${HEADERS} LINKDEF LinkDef.h)
 
# Generate the shared library from the sources
add_library(${MODULE_NAME} SHARED ${SOURCES} ${MODULE_NAME}Dict.cxx)
set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries( ${MODULE_NAME} PUBLIC ${ROOT_LIBS} )

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/Base)
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}Dict_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}Dict.rootmap
  DESTINATION lib  )


