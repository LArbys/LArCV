#ifndef __WIREMASK_CXX__
#define __WIREMASK_CXX__

#include "WireMask.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static WireMaskProcessFactory __global_WireMaskProcessFactory__;

  WireMask::WireMask(const std::string name)
    : ProcessBase(name)
  {}
    
  void WireMask::configure(const PSet& cfg)
  {
    _wire_v = cfg.get<std::vector<size_t> >("WireList");
    _image_producer = cfg.get<std::string>("ImageProducer");
    _plane_id = cfg.get<size_t>("PlaneID");
    _mask_val = cfg.get<float>("MaskValue",0);
  }

  void WireMask::initialize()
  {}

  bool WireMask::process(IOManager& mgr)
  {
    auto input_image_v = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));
    if(!input_image_v) {
      LARCV_CRITICAL() << "Invalid image producer name: " << _image_producer << std::endl;
      throw larbys();
    }

    // For operation, move an array to this scope
    std::vector<Image2D> image_v;
    input_image_v->Move(image_v);

    // make sure plane id is valid
    if(image_v.size() <= _plane_id) {
      LARCV_CRITICAL() << "Could not find plane: " << _plane_id << std::endl;
      throw larbys();
    }

    // get a handle on modifiable reference
    auto& img = image_v[_plane_id];

    // figure out compression factor used, and also prepare empty column to memcpy
    auto const compression_x = img.meta().width() / img.meta().cols();
    std::vector<float> empty_column(_mask_val,img.meta().rows());

    // Loop over wires, find target column and erase
    for(auto const& ch : _wire_v) {
      size_t target_col = (size_t)(ch / compression_x);
      img.copy(target_col,0,empty_column);
    }

    // put back an image
    input_image_v->Emplace(std::move(image_v));

    return true;
  }

  void WireMask::finalize()
  {}

}
#endif
