set(MODULE_NAME LArCVCoreDataFormat)

set( LIBNAME ${MODULE_NAME} )

# Collect the headers
set( HEADERS
BBox.h
ChStatus.h
ClusterMask.h
DataFormatTypes.h
DataFormatUtil.h
DataProductFactory.h
EventBase.h
EventChStatus.h
EventClusterMask.h
EventImage2D.h
EventPGraph.h
EventParticle.h
EventPixel2D.h
EventROI.h
EventSparseImage.h
EventVoxel2D.h
EventVoxel3D.h
Image2D.h
ImageMeta.h
PGraph.h
Particle.h
ParticleMass.h
Pixel2D.h
Pixel2DCluster.h
Point.h
ProductMap.h
ROI.h
SparseImage.h
UtilFunc.h
Vertex.h
Voxel.h
Voxel2D.h
Voxel3D.h
Voxel3DMeta.h
eigen.h
utils.h
IOManager.h )

add_library( ${LIBNAME} SHARED
BBox.cxx
ChStatus.cxx
ClusterMask.cxx
DataFormatUtil.cxx
DataProductFactory.cxx
EventBase.cxx
EventChStatus.cxx
EventClusterMask.cxx
EventImage2D.cxx
EventPGraph.cxx
EventParticle.cxx
EventPixel2D.cxx
EventROI.cxx
EventSparseImage.cxx
EventVoxel2D.cxx
EventVoxel3D.cxx
Image2D.cxx
ImageMeta.cxx
PGraph.cxx
Particle.cxx
ParticleMass.cxx
Pixel2DCluster.cxx
Point.cxx
ProductMap.cxx
ROI.cxx
SparseImage.cxx
UtilFunc.cxx
Vertex.cxx
Voxel.cxx
Voxel2D.cxx
Voxel3D.cxx
Voxel3DMeta.cxx
eigen.cxx
utils.cxx
IOManager.cxx
)

# set headers target property
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# set the include directories to reference during build
target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larcv/core/DataFormat>
  PRIVATE
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${MODULE_NAME} PUBLIC LArCVCoreBase ${ROOT_LIBRARIES})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} )

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)

# add dictionary file to sources
target_sources( ${MODULE_NAME} PRIVATE G__${MODULE_NAME}.cxx )

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
