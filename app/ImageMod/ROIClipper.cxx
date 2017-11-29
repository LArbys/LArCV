#ifndef __ROICLIPPER_CXX__
#define __ROICLIPPER_CXX__

#include "ROIClipper.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static ROIClipperProcessFactory __global_ROIClipperProcessFactory__;

  ROIClipper::ROIClipper(const std::string name)
    : ProcessBase(name)
  {}
    
  void ROIClipper::configure(const PSet& cfg)
  {
    _img_producer        = cfg.get<std::string>("ImageProducer");
    _input_roi_producer  = cfg.get<std::string>("InputROIProducer");
    _output_roi_producer = cfg.get<std::string>("OutputROIProducer");
  }

  void ROIClipper::initialize()
  {}

  bool ROIClipper::process(IOManager& mgr)
  {

    LARCV_DEBUG() << "done" << std::endl;

    const auto ev_img2d = (EventImage2D*)mgr.get_data(kProductImage2D,_img_producer);
    const auto ev_inroi = (EventROI*)mgr.get_data(kProductROI,_input_roi_producer);
    auto ev_outroi      = (EventROI*)mgr.get_data(kProductROI,_output_roi_producer);

    for (const auto& inroi  : ev_inroi->ROIArray()) {
      
      ROI outroi;
      
      for(size_t plane=0; plane<3; ++plane) {
	const auto& bb = inroi.BB(plane);
	const auto& img2d = ev_img2d->Image2DArray().at(plane);

	auto overlap_bb = img2d.meta().overlap(bb);
	outroi.AppendBB(overlap_bb);
      }

      ev_outroi->Emplace(std::move(outroi));
    }
    
    LARCV_DEBUG() << "end" << std::endl;

    return true;
  }

  void ROIClipper::finalize()
  {}

}
#endif
