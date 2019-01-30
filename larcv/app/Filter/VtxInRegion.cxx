#ifndef __VTXINREGION_CXX__
#define __VTXINREGION_CXX__

#include "VtxInRegion.h"

#include "DataFormat/EventROI.h"

namespace larcv {

  static VtxInRegionProcessFactory __global_VtxInRegionProcessFactory__;

  VtxInRegion::VtxInRegion(const std::string name)
    : ProcessBase(name)
  {}
    
  void VtxInRegion::configure(const PSet& cfg)
  {
    LARCV_INFO() << "start\n";
    auto pt1  = cfg.get<std::vector<double>>("YZPoint1");
    auto pt2  = cfg.get<std::vector<double>>("YZPoint2");
    _above = cfg.get<bool>("Above");
    _buffer = cfg.get<double>("Buffer");
    _roi_producer = cfg.get<std::string>("ROIProducer");
    
    if (pt1.size() != 2 or pt2.size() != 2) throw larbys("pt1 and pt2 not of size 2");
    
    _slope = (pt2[1] - pt1[1]) / ( pt2[0] - pt1[0]);

    _yinter = -1.0*_slope*pt2[0]+pt2[1];
    
    LARCV_INFO() << "_slope = "<<_slope<<" _yinter = "<<_yinter << std::endl;
    
    if (_slope == 0.0) throw larbys("Slope is zero");
    
    float angle=std::atan2(1.0,_slope);
    auto k = _buffer*std::sin(3.14159/2.0-angle);

    LARCV_INFO() << "k=" << k << "\n";
    LARCV_INFO() << "buffer = " << _buffer << std::endl;
    LARCV_INFO() << "angle  = " << angle << std::endl;
    LARCV_INFO() << "new _yinter = " << _yinter-k << std::endl;
    _yinter = _yinter-k;
  }
  
  void VtxInRegion::initialize()
  {
    LARCV_INFO() << "start";
  }
  
  bool VtxInRegion::process(IOManager& mgr)
  {
    LARCV_INFO() << "start" << std::endl;
    auto event_roi = (EventROI*)mgr.get_data(kProductROI,_roi_producer);

    const ROI* nu_roi;
    for (const auto& roi : event_roi->ROIArray()) {
      if (roi.Type() == 2) { 
	nu_roi = &roi; 
	break; 
      }
    }
    
    LARCV_INFO() << "Checking (Y,Z) : (" << nu_roi->Y() << "," << nu_roi->Z() << ")\n";
    if ( in_region(nu_roi->Y(),nu_roi->Z()) ) 
      { 
	LARCV_INFO() << "It's in the region return false to exclude\n";
	return false;
      }

    LARCV_INFO() << "return true" << std::endl;
    return true;
  }
  
  
  bool VtxInRegion::in_region(double Y,double Z) {
    
    bool in;
    
    if (_above) {
      in = Y > (_slope*Z + _yinter);
      if (in) return true;
      return false;
    }
    
    else {
      in = Y < _slope*Z + _yinter;
      if (in) return true;
      return false;
    }
    
  }
  
  void VtxInRegion::finalize()
  {
    LARCV_INFO() << "start" << std::endl;
  }

}
#endif
