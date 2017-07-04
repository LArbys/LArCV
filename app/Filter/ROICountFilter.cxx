#ifndef __ROICOUNTFILTER_CXX__
#define __ROICOUNTFILTER_CXX__

#include "ROICountFilter.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static ROICountFilterProcessFactory __global_ROICountFilterProcessFactory__;

  ROICountFilter::ROICountFilter(const std::string name)
    : ProcessBase(name)
  {}
    
  void ROICountFilter::configure(const PSet& cfg)
  {
    _roi_producer = cfg.get<std::string>("ROIProducer");
    _max_roi_count = cfg.get<size_t>("MaxROICount");
  }

  void ROICountFilter::initialize()
  {}

  bool ROICountFilter::process(IOManager& mgr)
  {
    auto ev_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_producer));
    if(!ev_roi) {
      LARCV_CRITICAL() << "ROI w/ label " << _roi_producer << " not found!" << std::endl;
      throw larbys();
    }
    
    auto const& roi_v = ev_roi->ROIArray();
    if(roi_v.size() >= _roi_count_v.size())
      _roi_count_v.resize(roi_v.size()+1,0);
    _roi_count_v[roi_v.size()] += 1;

    return (ev_roi->ROIArray().size() <= _max_roi_count);
  }

  void ROICountFilter::finalize()
  {
    double total_count = 0;
    for(auto const& v : _roi_count_v) total_count += v;
    LARCV_NORMAL() << "Reporting ROI counts. Total events processed: " << (int)total_count << std::endl;
    for(size_t i=0; i<_roi_count_v.size(); ++i) 
      LARCV_NORMAL() << "    Multi=" << i 
		     << " ... " << _roi_count_v[i] << " events (" 
		     << _roi_count_v[i] / total_count * 100 << " %)" << std::endl;
  }

}
#endif
