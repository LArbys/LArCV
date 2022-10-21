set(MODULE_NAME LArCVCoreDataFormat)

# DEFINE THE TARGET
set( LIBNAME ${MODULE_NAME} )

# Collect the headers
set( HEADERS
  BBox.h
  ChStatus.h
  ClusterMask.h
  DataFormatTypes.h
  DataFormatUtil.h
  DataProductFactory.h
  eigen.h
  EventBase.h
  EventChStatus.h
  EventClusterMask.h
  EventImage2D.h
  EventParticle.h
  EventPGraph.h
  EventPixel2D.h
  EventROI.h
  EventSparseImage.h
  EventVoxel2D.h
  EventVoxel3D.h
  EventSparseTensor2D.h  
  EventSparseTensor3D.h
  EventClusterVoxel3D.h    
  Image2D.h
  ImageMeta.h
  IOManager.h
  Particle.h
  ParticleMass.h
  PGraph.h
  Pixel2DCluster.h
  Pixel2D.h
  Point.h
  ProductMap.h
  ROI.h
  SparseImage.h
  UtilFunc.h
  utils.h
  Vertex.h
  Voxel2D.h
  Voxel3D.h
  SparseTensor2D.h  
  SparseTensor3D.h
  ClusterVoxel3D.h  
  Voxel3DMeta.h
  Voxel.h  
)

add_library( ${LIBNAME} SHARED
  BBox.cxx
  ChStatus.cxx
  ClusterMask.cxx
  DataFormatUtil.cxx
  DataProductFactory.cxx
  eigen.cxx
  EventBase.cxx
  EventChStatus.cxx
  EventClusterMask.cxx
  EventImage2D.cxx
  EventParticle.cxx
  EventPGraph.cxx
  EventPixel2D.cxx
  EventROI.cxx
  EventSparseImage.cxx
  EventVoxel2D.cxx
  EventVoxel3D.cxx
  EventSparseTensor2D.cxx  
  EventSparseTensor3D.cxx
  EventClusterVoxel3D.cxx  
  Image2D.cxx
  ImageMeta.cxx
  IOManager.cxx
  Particle.cxx
  ParticleMass.cxx
  PGraph.cxx
  Pixel2DCluster.cxx
  Point.cxx
  ProductMap.cxx
  ROI.cxx
  SparseImage.cxx
  UtilFunc.cxx
  utils.cxx
  Vertex.cxx
  Voxel2D.cxx
  Voxel3D.cxx
  SparseTensor2D.cxx  
  SparseTensor3D.cxx
  ClusterVoxel3D.cxx  
  Voxel3DMeta.cxx
  Voxel.cxx  
)

# Generate the dictionary
# headers for rootcling
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} )
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)
target_sources( ${LIBNAME} PRIVATE G__${MODULE_NAME}.cxx )

set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

#message("hmm: ${PROJECT_SOURCE_DIR}")
#message("hmm:   $<INSTALL_INTERFACE:include/>")
target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larcv/core/DataFormat>
  ${PYTHON_INCLUDE_PATH}
  ${ROOT_INCLUDE_DIRS}
  PRIVATE
  ${PROJECT_SOURCE_DIR}  
#  ${EIGEN3_INCLUDE_DIRS} )
)

target_link_libraries(${MODULE_NAME}
  PUBLIC
  LArCVCoreBase
  ${ROOT_LIBRARIES})

# install libraries and headers
install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/core/DataFormat)

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )
