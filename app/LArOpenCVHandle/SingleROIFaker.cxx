#ifndef __SINGLEROIFAKER_CXX__
#define __SINGLEROIFAKER_CXX__

#include "SingleROIFaker.h"

#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static SingleROIFakerProcessFactory __global_SingleROIFakerProcessFactory__;

  SingleROIFaker::SingleROIFaker(const std::string name)
    : ProcessBase(name)
  {}
    
  void SingleROIFaker::configure(const PSet& cfg)
  {
    _img_producer = cfg.get<std::string>("ImageProducer");
    _roi_producer = cfg.get<std::string>("ROIProducer");
  }

  void SingleROIFaker::initialize()
  {}

  bool SingleROIFaker::process(IOManager& mgr)
  {
    const auto ev_img = (EventImage2D*)mgr.get_data(kProductImage2D,_img_producer);
    auto ev_roi = (EventROI*)mgr.get_data(kProductROI,_roi_producer);
    
    if (!ev_roi->ROIArray().empty())
      return true;

    larcv::ROI proi;
    for(const auto& img : ev_img->Image2DArray())
      proi.AppendBB(img.meta());
    
    ev_roi->Emplace(std::move(proi));
    return true;
  }

  void SingleROIFaker::finalize()
  {}

}
#endif
