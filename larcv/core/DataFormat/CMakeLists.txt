# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")
 
# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(LArCVCoreDataFormatDict ${HEADERS} LINKDEF LinkDef.h)
 
# Generate the shared library from the sources
add_library(LArCVCoreDataFormat SHARED ${SOURCES} LArCVCoreDataFormatDict.cxx)
set_target_properties(LArCVCoreDataFormat PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_include_directories( LArCVCoreDataFormat PUBLIC ${LARCV_SOURCE_DIR} )

install(TARGETS LArCVCoreDataFormat
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/DataFormat)


