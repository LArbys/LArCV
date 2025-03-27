set(MODULE_NAME LArCVCoreCVUtil)

# Collect the headers
set( HEADERS 
  load_cvutil.h
  CVUtil.h
)

# Collect the sources
add_library( ${MODULE_NAME} SHARED 
  load_cvutil.cxx
  CVUtil.cxx
)

set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_include_directories(${MODULE_NAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larcv/core/CVUtil>
  PRIVATE
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries( ${MODULE_NAME} PUBLIC 
  LArCVCoreBase 
  LArCVCoreDataFormat 
  opencv_core 
  opencv_imgcodecs
)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR} 
  ${OpenCV_INCLUDE_DIRS} 
)
# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)

# add dictionary file to sources
target_sources( ${MODULE_NAME} PRIVATE G__${MODULE_NAME}.cxx )

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/CVUtil)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )
