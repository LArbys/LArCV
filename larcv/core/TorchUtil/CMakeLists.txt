set(MODULE_NAME LArCVCoreTorchUtil)

# Collect the headers
set( HEADERS 
  TorchUtils.h
)

add_library( ${MODULE_NAME} SHARED 
  TorchUtils.cxx
)

set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_include_directories(${MODULE_NAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larcv/core/TorchUtil>
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries( ${MODULE_NAME} PUBLIC 
  LArCVCoreBase 
  LArCVCoreDataFormat
  ${TORCH_LIBRARIES}
)
include_directories( 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR} )

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/TorchUtil)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )
