set(MODULE_NAME LArCVCoreDataFormat)

# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")
 
# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

# includes
#message("Base dir: ${LARCV_BASE_DIR}")
include_directories( ${LARCV_BASE_DIR} )

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(LArCVCoreDataFormatDict ${HEADERS} LINKDEF LinkDef.h)
 
# Generate the shared library from the sources
add_library(LArCVCoreDataFormat SHARED ${SOURCES} LArCVCoreDataFormatDict.cxx)
set_target_properties(LArCVCoreDataFormat PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries( LArCVCoreDataFormat PUBLIC LArCVCoreBase )

# install libraries and headers
install(TARGETS LArCVCoreDataFormat
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/DataFormat)

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}Dict_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}Dict.rootmap
  DESTINATION lib  )
