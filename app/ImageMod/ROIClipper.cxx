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
    _remove_duplicates   = cfg.get<bool>("RemoveDuplicates");
  }

  void ROIClipper::initialize()
  {}

  bool ROIClipper::process(IOManager& mgr)
  {

    LARCV_DEBUG() << "done" << std::endl;

    const auto ev_img2d = (EventImage2D*)mgr.get_data(kProductImage2D,_img_producer);
    const auto ev_inroi = (EventROI*)mgr.get_data(kProductROI,_input_roi_producer);
    auto ev_outroi      = (EventROI*)mgr.get_data(kProductROI,_output_roi_producer);

    std::vector<ROI> tmp_outroi_v;
    tmp_outroi_v.reserve(ev_inroi->ROIArray().size());
    
    for (const auto& inroi  : ev_inroi->ROIArray()) {
      tmp_outroi_v.resize(tmp_outroi_v.size()+1);
      auto& outroi = tmp_outroi_v.back();
  
      for(size_t plane=0; plane<3; ++plane) {
	const auto& bb = inroi.BB(plane);
	const auto& img2d = ev_img2d->Image2DArray().at(plane);
	
	auto overlap_bb = img2d.meta().overlap(bb);
	outroi.AppendBB(overlap_bb);
      }
    }


    if(!_remove_duplicates) {
      ev_outroi->Emplace(std::move(tmp_outroi_v));    
      LARCV_DEBUG() << "end" << std::endl;
      return true;
    }
    
    std::vector<ROI> outroi_v;
    outroi_v.reserve(tmp_outroi_v.size());
    
    std::vector<bool> used_v(tmp_outroi_v.size(),false);
    std::vector<bool> dup_v(tmp_outroi_v.size(),false);

    for(size_t rid1=0; rid1<tmp_outroi_v.size(); ++rid1) {
      if (used_v[rid1]) continue;
      auto& roi1 = tmp_outroi_v[rid1];

      bool unique = true;      

      for(size_t rid2=rid1+1; rid2<tmp_outroi_v.size(); ++rid2) {
	if (used_v[rid2]) continue;
	auto& roi2 = tmp_outroi_v[rid2];

	bool dup = true;

	for(size_t plane=0; plane<3; ++plane)
	  dup &= (roi1.BB(plane) == roi2.BB(plane));

	dup_v[rid2] = dup;

	if(dup) unique  = false;
      }
      
      outroi_v.emplace_back(std::move(roi1));

      used_v[rid1] = true;

      if (!unique) {
	for(size_t dupid=0; dupid<dup_v.size(); ++dupid) {
	  if (!dup_v[dupid]) continue;
	  used_v[dupid] = true;
	}
      }
      
      std::fill(dup_v.begin(), dup_v.end(), false);
    }
    
    ev_outroi->Emplace(std::move(outroi_v));    
    LARCV_DEBUG() << "end" << std::endl;
    return true;
  }

  void ROIClipper::finalize()
  {}

}
#endif
