#ifndef __MULTIROICROPPER_H__
#define __MULTIROICROPPER_H__

#include "WholeImageCropper.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static WholeImageCropperProcessFactory __global_WholeImageCropperProcessFactory__;

  WholeImageCropper::WholeImageCropper(const std::string name)
    : ProcessBase(name)
  {}
    
  void WholeImageCropper::configure(const PSet& cfg)
  {
    _image_producer = cfg.get<std::string>("ImageProducer");
    _target_rows = cfg.get<size_t>("TargetRows");
    _target_cols = cfg.get<size_t>("TargetCols");
    _target_ch   = cfg.get<size_t>("TargetChannel");
  }

  void WholeImageCropper::initialize()
  {}

  bool WholeImageCropper::process(IOManager& mgr)
  {
    _cropped_v.clear();

    // assert valid pointer
    auto event_img_v = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));
    if(!event_img_v){
      LARCV_CRITICAL() << "EventImage2D not found for label " << _image_producer << std::endl;
      throw larbys();
    }

    // assert target image ch
    auto const& img_v = event_img_v->Image2DArray();
    if(img_v.size() <= _target_ch) {
      LARCV_CRITICAL() << "EventImage2D size " << img_v.size() << " <= target channel " << _target_ch << std::endl;
      throw larbys();
    }

    // get image, meta
    _image = img_v[_target_ch];
    auto const& meta = _image.meta();
    LARCV_INFO() << "Image dimension: " << std::endl << meta.dump() << std::endl;
    double px_width  = meta.width()  / (double)(meta.cols());
    double px_height = meta.height() / (double)(meta.rows());
    double min_x = meta.min_x();
    double min_y = meta.min_y();
    double max_x = meta.max_x();
    double max_y = meta.max_y();
    size_t max_col = meta.cols();
    size_t max_row = meta.rows();

    size_t nbox_x = max_col / _target_cols + (max_col % _target_cols > 0 ? 1 : 0);
    size_t nbox_y = max_row / _target_rows + (max_row % _target_rows > 0 ? 1 : 0);

    std::vector<larcv::ImageMeta> meta_v;    
    for(size_t ix=0; ix < nbox_x; ++ix) {
      
      double crop_min_x = min_x + ix     * _target_cols * px_width;
      double crop_max_x = min_x + (ix+1) * _target_cols * px_width;
      if(crop_max_x > max_x) 
	crop_min_x = max_x - _target_cols * px_width;
      
      for(size_t iy=0; iy < nbox_y; ++iy) {
	double crop_max_y = max_y - iy     * _target_rows * px_height;
	double crop_min_y = max_y - (iy+1) * _target_rows * px_height;
	if(crop_min_y < min_y)
	  crop_max_y = min_y + _target_rows * px_height;
	
	meta_v.push_back(ImageMeta(px_width * _target_cols, px_height * _target_rows,
				   _target_rows, _target_cols,
				   crop_min_x, crop_max_y,
				   meta.plane()));
	LARCV_INFO() << "Small ROI " << meta_v.size() - 1 << ": " << std::endl
		     << meta_v.back().dump() << std::endl;
      }
    }

    _cropped_v.clear();
    for(auto const& meta : meta_v)
      _cropped_v.emplace_back(std::move(_image.crop(meta)));

    return true;
  }

  void WholeImageCropper::finalize()
  {}

}
#endif
