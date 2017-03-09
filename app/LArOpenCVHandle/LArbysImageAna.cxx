#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name),
      _LArbysImageMaker()
  {}
  
  void LArbysImageAna::configure(const PSet& cfg)
  {
    _adc_producer = cfg.get<std::string>("ADCProducer");
    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
    
  }

  void LArbysImageAna::initialize()
  {}

  bool LArbysImageAna::process(IOManager& mgr)
  {
    auto img_data_v = _LArbysImageMaker.ExtractImage(mgr,_adc_producer);

    
  }

  void LArbysImageAna::finalize()
  {}

}
#endif
