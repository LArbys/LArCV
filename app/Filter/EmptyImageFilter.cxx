#ifndef __EMPTYIMAGEFILTER_CXX__
#define __EMPTYIMAGEFILTER_CXX__

#include "EmptyImageFilter.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static EmptyImageFilterProcessFactory __global_EmptyImageFilterProcessFactory__;

  EmptyImageFilter::EmptyImageFilter(const std::string name)
    : ProcessBase(name)
  {}
    
  void EmptyImageFilter::configure(const PSet& cfg)
  {
    _image_producer = cfg.get<std::string>("ImageProducer");
  }

  void EmptyImageFilter::initialize()
  {}

  bool EmptyImageFilter::process(IOManager& mgr)
  {
    auto ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));
    if(!ev_image || ev_image->Image2DArray().empty()) return false;
    return true;
  }

  void EmptyImageFilter::finalize()
  {}

}
#endif
