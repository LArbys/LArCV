#ifndef __SUPERAWIRE_CXX__
#define __SUPERAWIRE_CXX__

#include "SuperaWire.h"
#include "LAr2Image.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static SuperaWireProcessFactory __global_SuperaWireProcessFactory__;

  SuperaWire::SuperaWire(const std::string name)
    : SuperaBase(name)
  {}

  void SuperaWire::configure(const PSet& cfg)
  { SuperaBase::configure(cfg); }

  void SuperaWire::initialize()
  { SuperaBase::initialize(); }

  bool SuperaWire::process(IOManager& mgr)
  {
    SuperaBase::process(mgr);
	
    auto const& meta_v = Meta();
    
    if(meta_v.empty()) {
      LARCV_CRITICAL() << "Meta not created!" << std::endl;
      throw larbys();
    }
    auto ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,OutImageLabel()));
    if(!ev_image) {
      LARCV_CRITICAL() << "Output image could not be created!" << std::endl;
      throw larbys();
    }
    if(!(ev_image->Image2DArray().empty())) {
      LARCV_CRITICAL() << "Output image array not empty!" << std::endl;
      throw larbys();
    }

    auto image_v = supera::Wire2Image2D(meta_v, LArData<supera::LArWire_t>(), TimeOffset());

    for(size_t plane=0; plane<image_v.size(); ++plane) {
      auto& image = image_v[plane];
      image.compress(image.meta().rows() / RowCompressionFactor().at(plane),
		     image.meta().cols() / ColCompressionFactor().at(plane),
		     larcv::Image2D::kSum);
    }

    ev_image->Emplace(std::move(image_v));
    return true;
  }

  void SuperaWire::finalize()
  {}


}
#endif
