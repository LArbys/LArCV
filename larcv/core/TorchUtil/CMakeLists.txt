set(MODULE_NAME LArCVCoreTorchUtil)

# Collect the headers
file(GLOB HEADERS "*.h")

# Remove LinkDef.h
list(FILTER HEADERS EXCLUDE REGEX ".*LinkDef.h$")
 
# However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "*.cxx")

if (Torch_FOUND)
  include_directories ( ${LARCV_BASE_DIR} ${TORCH_INCLUDE_DIRS} )
endif()

set(CMAKE_CXX_STANDARD 17)

# Generate the dictionary
#ROOT_GENERATE_DICTIONARY( G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)
 
# Generate the shared library from the sources
#add_library(${MODULE_NAME} SHARED ${SOURCES} G__${MODULE_NAME}.cxx)
add_library(${MODULE_NAME} SHARED ${SOURCES})
set_target_properties( ${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
if (Torch_FOUND)
   message("torch libs: ${TORCH_LIBRARIES}")
   target_link_libraries( ${MODULE_NAME} PRIVATE ${TORCH_LIBRARIES} LArCVCoreBase LArCVCoreDataFormat )
else()
   target_link_libraries( ${MODULE_NAME} PUBLIC LArCVCoreBase LArCVCoreDataFormat )
endif()

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/TorchUtil)

#install( FILES
#  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
#  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
#  DESTINATION lib  )
