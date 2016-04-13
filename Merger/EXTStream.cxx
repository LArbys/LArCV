#ifndef __EXTSTREAM_CXX__
#define __EXTSTREAM_CXX__

#include "EXTStream.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static EXTStreamProcessFactory __global_EXTStreamProcessFactory__;

  EXTStream::EXTStream(const std::string name)
    : ImageHolder(name)
  {}
    
  void EXTStream::configure(const PSet& cfg)
  {
    _image_producer = cfg.get<std::string>("ImageProducer");
  }

  void EXTStream::initialize()
  {}

  bool EXTStream::process(IOManager& mgr)
  {
    _image_v.clear();

    auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));

    if(!event_image || event_image->Image2DArray().empty()) return false;

    for(auto const& img : event_image->Image2DArray())
      
      _image_v.push_back(img);

    retrieve_id(event_image);

    return true;
  }

  void EXTStream::finalize(TFile* ana_file)
  {}

}
#endif
