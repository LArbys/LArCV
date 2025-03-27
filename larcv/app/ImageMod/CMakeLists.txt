set(MODULE_NAME LArCVAppImageMod)

# Collect the headers
set( HEADERS 
  ADCScale.h
  ADCScaleAna.h
  ADCThreshold.h
  Binarize.h
  ChannelMax.h
  CombineImages.h
  Compressor.h
  CosmicROIFiller.h
  CropROI.h
  EmbedImage.h
  ImageChargeAna.h
  ImageFromPixel2D.h
  ImageFromPixel2DCluster.h
  ImageModUtils.h
  LoadImageMod.h
  MaskImage.h
  MeanSubtraction.h
  ModularPadImage.h
  MultiROICropper.h
  PlaneImageMerger.h
  ROIClipper.h
  ROIMask.h
  ROIMerger.h
  ROIPad.h
  ResizeImage.h
  SegWeightInstance.h
  SegWeightTrackShower.h
  SegmentAna.h
  SegmentMaker.h
  SegmentMask.h
  SegmentRelabel.h
  SimpleDigitizer.h
  SliceImages.h
  StepDigitizer.h
  UnionROI.h
  WireMask.h
)

set( SOURCES 
  ADCScale.cxx
  ADCScaleAna.cxx
  ADCThreshold.cxx
  Binarize.cxx
  ChannelMax.cxx
  CombineImages.cxx
  Compressor.cxx
  CosmicROIFiller.cxx
  CropROI.cxx
  EmbedImage.cxx
  ImageChargeAna.cxx
  ImageFromPixel2D.cxx
  ImageFromPixel2DCluster.cxx
  ImageModUtils.cxx
  LoadImageMod.cxx
  MaskImage.cxx
  MeanSubtraction.cxx
  ModularPadImage.cxx
  MultiROICropper.cxx
  PlaneImageMerger.cxx
  ROIClipper.cxx
  ROIMask.cxx
  ROIMerger.cxx
  ROIPad.cxx
  ResizeImage.cxx
  SegWeightInstance.cxx
  SegWeightTrackShower.cxx
  SegmentAna.cxx
  SegmentMaker.cxx
  SegmentMask.cxx
  SegmentRelabel.cxx
  SimpleDigitizer.cxx
  SliceImages.cxx
  StepDigitizer.cxx
  UnionROI.cxx
  WireMask.cxx
)

# We remove sources for those classes with non-standard dependencies
if(NOT LARCV_USE_GEO2D)
  message(STATUS "removing classes with Geo2D dependenies from ImageMod source list")
  list(FILTER SOURCES EXCLUDE REGEX ".*ROIMerger.cxx$")
  list(FILTER HEADERS EXCLUDE REGEX ".*ROIMerger.h$")
endif()

add_library( ${MODULE_NAME} SHARED
  ${SOURCES}
)

set_target_properties(${MODULE_NAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# set the include directories to reference during build
target_include_directories(${MODULE_NAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larcv/core/Base>
  PRIVATE
  ${PROJECT_SOURCE_DIR}
)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} )
if(LARCV_USE_GEO2D)
  include_directories( ${GEO2D_INC_DIR} ${OpenCV_INCLUDE_DIRS} )
endif()

target_link_libraries( ${MODULE_NAME} PUBLIC 
  ${ROOT_LIBS}
  LArCVCoreBase 
  LArCVCoreDataFormat 
  LArCVCoreProcessor
)
if (LARCV_USE_GEO2D)
  target_link_libraries( ${MODULE_NAME} PUBLIC ${GEO2D_LIBS} ${OpenCV_LIBS} )
endif()

# Generate the dictionary
ROOT_GENERATE_DICTIONARY(G__${MODULE_NAME} ${HEADERS} LINKDEF LinkDef.h)
 
# add dictionary file to sources
target_sources( ${MODULE_NAME} PRIVATE G__${MODULE_NAME}.cxx )

# install libraries and headers
install(TARGETS ${MODULE_NAME}
  EXPORT larcv
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larcv/app/ImageMod)

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${MODULE_NAME}.rootmap
  DESTINATION lib  )
