set(MODULE_NAME LArCVAppImageMod)

# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")
 
# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

# We remove sources for those classes with non-standard dependencies
if(NOT LARCV_USE_GEO2D)
  message("removing classes with Geo2D dependenies from ImageMod source list")
  list(FILTER SOURCES EXCLUDE REGEX ".*ROIMerger.cxx$")
  list(FILTER HEADERS EXCLUDE REGEX ".*ROIMerger.h$")
endif()

# includes
include_directories( ${LARCV_BASE_DIR} )
if(LARCV_USE_GEO2D)
  include_directories( ${GEO2D_INC_DIR} )
endif()

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(${MODULE_NAME}Dict ${HEADERS} LINKDEF LinkDef.h)
 
# Generate the shared library from the sources
add_library(${MODULE_NAME} SHARED ${SOURCES} ${MODULE_NAME}Dict.cxx)
set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries(${MODULE_NAME} PUBLIC LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor)
target_link_libraries(${MODULE_NAME} PUBLIC ${GEO2D_LIBS})

# install libraries and headers
install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/app/ImageMod)

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}Dict_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}Dict.rootmap
  DESTINATION lib  )
