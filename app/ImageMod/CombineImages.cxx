#ifndef __COMBINEIMAGES_CXX__
#define __COMBINEIMAGES_CXX__

#include "CombineImages.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static CombineImagesProcessFactory __global_CombineImagesProcessFactory__;

  CombineImages::CombineImages(const std::string name)
    : ProcessBase(name)
  {}
    
  void CombineImages::configure(const PSet& cfg)
  {
    _producer_v = cfg.get<std::vector<std::string> >("ImageProducers");
    _nplanes    = cfg.get<size_t>("NPlanes");
    _out_producer = cfg.get<std::string>("OutputProducer");
  }

  void CombineImages::initialize()
  {}

  bool CombineImages::process(IOManager& mgr)
  {
    
    std::vector<larcv::Image2D> image_v;
    image_v.resize(_nplanes * _producer_v.size());
    for(size_t i=0; i<_producer_v.size(); ++i) {

      auto const& producer = _producer_v[i];

      auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,producer));

      if(event_image->Image2DArray().size()!=_nplanes) {
	LARCV_CRITICAL() << "Producer " << producer 
			 << " has # images " << event_image->Image2DArray().size() 
			 << " != # planes " << _nplanes << std::endl;
	throw larbys();
      }

      std::vector<larcv::Image2D> images;
      event_image->Move(images);

      for(size_t plane=0; plane<_nplanes; ++plane)

	image_v[plane*_producer_v.size()+i] = std::move(images[plane]);
      
    }

    auto out_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_out_producer));

    out_image->Emplace(std::move(image_v));

    return true;
  }

  void CombineImages::finalize()
  {}

}
#endif
