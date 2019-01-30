# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")
 
# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(LArCVCoreBaseDict ${HEADERS} LINKDEF LinkDef.h)
 
# Generate the shared library from the sources
add_library(LArCVCoreBase SHARED ${SOURCES} LArCVCoreBaseDict.cxx)
set_target_properties(LArCVCoreBase PROPERTIES PUBLIC_HEADER "${HEADERS}")

install(TARGETS LArCVCoreBase
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/Base)
#install(FILES ${HEADERS} DESTINATION  include/larcv/core/Base)


