set(MODULE_NAME LArCVCoreBase)

set( LIBNAME ${MODULE_NAME} )

# Collect the headers
set( HEADERS
  LArCVTypes.h
  larcv_logger.h
  larcv_base.h
  larbys.h
  stack_trace.h # not needed by dict
  Watch.h # not needed by dict
  Parser.h
  PSet.h
  Range.h
  UniqueRangeSet.h
  LArCVBaseUtilFunc.h  
  ConfigManager.h
)

# set( DICT_HEADERS 
#   LArCVTypes.h
#   larcv_logger.h
#   larcv_base.h
#   larbys.h
#   Parser.h
#   PSet.h
#   Range.h # maybe can avoid these?
#   UniqueRangeSet.h # maybe can void parsing these?
#   LArCVBaseUtilFunc.h  
#   ConfigManager.h
# )

add_library( ${LIBNAME} SHARED
  larcv_logger.cxx
  Parser.cxx
  PSet.cxx
  LArCVBaseUtilFunc.cxx
  ConfigManager.cxx
  Range.cxx
  UniqueRangeSet.cxx
)

set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# set the include directories to reference during build
target_include_directories(${MODULE_NAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larcv/core/Base>
  PRIVATE
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${MODULE_NAME} PUBLIC ${ROOT_LIBS} )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} )

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)
 
# add dictionary file to sources
target_sources( ${MODULE_NAME} PRIVATE G__${MODULE_NAME}.cxx )

install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/Base)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )
