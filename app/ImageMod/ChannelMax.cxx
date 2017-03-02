#ifndef __CHANNELMAX_CXX__
#define __CHANNELMAX_CXX__

#include "ChannelMax.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static ChannelMaxProcessFactory __global_ChannelMaxProcessFactory__;

  ChannelMax::ChannelMax(const std::string name)
    : ProcessBase(name)
  {}
    
  void ChannelMax::configure(const PSet& cfg)
  {
    _in_producer  = cfg.get<std::string>("InProducer");
    _nplanes      = cfg.get<size_t>("NPlanes");
    _out_producer = cfg.get<std::string>("OutputProducer");
  }

  void ChannelMax::initialize()
  {}

  bool ChannelMax::process(IOManager& mgr)
  {

    auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_in_producer));    
    if(!event_image) throw larbys("No event image found");
    auto& img_v = event_image->Image2DArray();
    
    larcv::Image2D image(img_v[0].meta());
    
    for(size_t row=0;row<image.meta().rows();++row){
      for(size_t col=0;col<image.meta().cols();++col){
	float maxpx(-1),maxpl(-1);
	for(size_t plane_id=0;plane_id<_nplanes;++plane_id) {
	  auto px=img_v[plane_id].pixel(row,col);
	  //px*=_plane_weight_v[plane_id];
	  if (px>maxpx) { maxpx=px; maxpl=plane_id; }
	}
	if (maxpx<0 or maxpl<0) throw larbys("No max plan identified");
	image.set_pixel(row,col,maxpl);
      }
    }
    
    auto out_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_out_producer));
    
    out_image->Emplace(std::move(image));

    return true;
  }

  void ChannelMax::finalize()
  {}

}
#endif
