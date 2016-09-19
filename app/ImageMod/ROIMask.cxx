#ifndef __ROIMASK_CXX__
#define __ROIMASK_CXX__

#include "ROIMask.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static ROIMaskProcessFactory __global_ROIMaskProcessFactory__;

  ROIMask::ROIMask(const std::string name)
    : ProcessBase(name)
  {}
    
  void ROIMask::configure(const PSet& cfg)
  {
    fInputImageProducer  = cfg.get<std::string>( "InputImageProducer" );
    fOutputImageProducer = cfg.get<std::string>( "OutputImageProducer" );
    fInputROIProducer    = cfg.get<std::string>( "InputROIProducer" );
    fMaskOutsideROI      = cfg.get<bool>( "MaskOutsideROI" );

    if (fInputImageProducer==fOutputImageProducer) finplace = true;
    else finplace = false;
      
  }

  void ROIMask::initialize()
  {}

  bool ROIMask::process(IOManager& mgr)
  {

    // Get images (and setup output)
    EventImage2D* event_original = (EventImage2D*)mgr.get_data(kProductImage2D, fInputImageProducer);
    EventImage2D* event_masked   = NULL;
    if (!finplace)
      event_masked  = (EventImage2D*)mgr.get_data(kProductImage2D, fOutputImageProducer);
    else
      event_masked  = event_original;

    if ( !event_original ) {
      LARCV_ERROR() << "Could not open input images." << std::endl;
      return false;
    }
    if ( !event_masked )  {
      LARCV_ERROR() << "Could not open output image holder." << std::endl;
      return false;      
    }

    // Get ROI
    EventROI* event_roi = (EventROI*)mgr.get_data(kProductROI, fInputROIProducer);
    auto rois = event_roi->ROIArray();
    int roi_id = 0;

    // original moves ownership of its image vectors to us
    std::vector< larcv::Image2D> image_v;
    event_original->Move(image_v ); 
 
    // container for new images if we are not working in place
    std::vector< larcv::Image2D > masked_image_v; 

    std::vector<float> _buffer;
    for ( size_t i=0; i<image_v.size(); ++i ) {

      // get access to the data
      larcv::Image2D& original_image = image_v[i];

      auto meta = original_image.meta();
      auto roi = rois.at(roi_id);
      auto bb  = roi.BB( meta.plane() );

      // make roi meta with same scale as input image
      double origin_x = bb.min_x();
      double origin_y = bb.max_y();
      double width    = bb.width();
      double height   = bb.height();

      // get row and col size based on scale of original image
      int cols = (int)width/meta.pixel_width();
      int rows = (int)height/meta.pixel_height();

      ImageMeta maskedmeta( width, height, rows, cols, origin_x, origin_y, meta.plane() );

      if ( fMaskOutsideROI ) {
	// we want everything but the ROI to be zero

	// we crop out the region we're interested in
	larcv::Image2D cropped_image = original_image.crop( maskedmeta );
      
	// define the new image, and blank it out
	larcv::Image2D new_image( meta );
	new_image.paint( 0.0 );
	// overlay cropped image
	new_image.overlay( cropped_image );
	masked_image_v.emplace_back( std::move(new_image) );
      }
      else {
	// we want to zero the region inside the ROI

	// make blank with size of ROi
	larcv::Image2D roi_blank( maskedmeta );
	roi_blank.paint(0.0);

	// make copy of original
	larcv::Image2D new_image( meta, original_image.as_vector() );
	new_image.overlay( roi_blank );
	masked_image_v.emplace_back( std::move(new_image) );
      }
      
    }

    // store
    if ( finplace ) {
      // give the new images
      event_original->Emplace( std::move(masked_image_v) );
    }
    else {
      // return the original images
      event_original->Emplace( std::move( image_v ) );
      // we give the new image2d container our relabeled images
      event_masked->Emplace( std::move(masked_image_v) );
    }


  }

  void ROIMask::finalize()
  {}

}
#endif
