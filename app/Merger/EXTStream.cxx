#ifndef __EXTSTREAM_CXX__
#define __EXTSTREAM_CXX__

#include "EXTStream.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventChStatus.h"

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

    // Retrieve ChStatus
    _chstatus_m.clear();
    
    auto event_chstatus = (EventChStatus*)(mgr.get_data(kProductChStatus,_image_producer));
    
    if(!event_chstatus || event_chstatus->ChStatusMap().empty()) return false;

    _chstatus_m = event_chstatus->ChStatusMap();

    // Retrieve Image
    _image_v.clear();

    auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));

    if(!event_image || event_image->Image2DArray().empty()) return false;

    for(auto const& img : event_image->Image2DArray())
      
      _image_v.push_back(img);

    // Retrieve event id
    retrieve_id(event_image);

    return true;
  }

  void EXTStream::finalize(TFile* ana_file)
  {}

}
#endif
