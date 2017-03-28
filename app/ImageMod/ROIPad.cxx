#ifndef __ROIPAD_CXX__
#define __ROIPAD_CXX__

#include "ROIPad.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static ROIPadProcessFactory __global_ROIPadProcessFactory__;

  ROIPad::ROIPad(const std::string name)
    : ProcessBase(name)
  {}
    
  void ROIPad::configure(const PSet& cfg)
  {
    _img_producer = cfg.get<std::string>("ImageProducer");
    _input_roi_producer = cfg.get<std::string>("InputROIProducer");
    _output_roi_producer = cfg.get<std::string>("OutputROIProducer");

    _row_pad=_col_pad=kINVALID_FLOAT;
    _row_pad = cfg.get<float>("PadRows");
    _col_pad = cfg.get<float>("PadCols");
  }
  
  void ROIPad::initialize()
  {}

  bool ROIPad::process(IOManager& mgr)
  {
    const auto evimg2d = (EventImage2D*)mgr.get_data(kProductImage2D,_img_producer);
    const auto inroi = (EventROI*)mgr.get_data(kProductROI,_input_roi_producer);
    auto ouroi = (EventROI*)mgr.get_data(kProductROI,_output_roi_producer);
    
    for (auto roi_pad : inroi->ROIArray()) {
      auto bb_v = roi_pad.BB();
      bb_v.clear();
      bb_v.reserve(roi_pad.BB().size());
      for( auto bb : roi_pad.BB() ) {
	ImageMeta bb_copy(bb.width()  + _col_pad*bb.pixel_width(),
			  bb.height() + _row_pad*bb.pixel_height(),
			  bb.rows()   + _row_pad,
			  bb.cols()   + _col_pad,
			  bb.tl().x - _col_pad,
			  bb.tl().y - _row_pad,
			  bb.plane());
	bool outside=false;
	const auto& meta = evimg2d->Image2DArray().at(bb.plane()).meta();
	if (bb_copy.max_x() > meta.max_x()) outside=true;
	if (bb_copy.max_y() > meta.max_y()) outside=true;
	if (bb_copy.min_x() < meta.min_x()) outside=true;
	if (bb_copy.min_y() < meta.min_y()) outside=true;
	    
	if (!outside) bb_v.emplace_back(std::move(bb_copy));
	else          bb_v.emplace_back(std::move(bb));
	
      }
      roi_pad.SetBB(bb_v);
      ouroi->Emplace(std::move(roi_pad));
    }
    
    return true;
  }

  void ROIPad::finalize()
  {}

}
#endif
