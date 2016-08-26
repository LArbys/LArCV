#ifndef __CROPROI_CXX__
#define __CROPROI_CXX__

#include "CropROI.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static CropROIProcessFactory __global_CropROIProcessFactory__;

  CropROI::CropROI(const std::string name)
    : ProcessBase(name)
  {}
    
  void CropROI::configure(const PSet& cfg)
  {
    _roi_producer = cfg.get<std::string>("ROIProducer");
    _image_producer = cfg.get<std::string>("ImageProducer");
  }

  void CropROI::initialize()
  {}

  bool CropROI::process(IOManager& mgr)
  {
    auto event_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_producer));
    if(!event_roi) {
      LARCV_CRITICAL() << "No ROI found with a name: " << _roi_producer << std::endl;
      throw larbys();
    }

    auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));
    if(!event_image) {
      LARCV_CRITICAL() << "No Image2D found with a name: " << _image_producer << std::endl;
      throw larbys();
    }

    auto const& roi_v = event_roi->ROIArray();

    // Make sure # ROI is 1, and holds ImageMeta per image
    if(roi_v.size() != 1) {
      LARCV_CRITICAL() << "EventROI has " << roi_v.size() << " entries (>1!)" << std::endl;
      throw larbys();
    }
    auto const& bb_v = roi_v[0].BB();
    std::vector<larcv::Image2D> image_v;
    event_image->Move(image_v);
    
    if(bb_v.size() != image_v.size()) {
      LARCV_CRITICAL() << "EventImage2D entry count (" << image_v.size()
		       << ") does not match with bounding-box count (" << bb_v.size()
		       << ") stored in ROI..." << std::endl;
      throw larbys();
    }

    // Now process
    for(size_t plane=0; plane<bb_v.size(); ++plane) {
      auto const& bb = bb_v[plane];
      auto& image = image_v[plane];
      image = image.crop(bb);
    }
    event_image->Emplace(std::move(image_v));
    return true;
  }

  void CropROI::finalize()
  {}

}
#endif
