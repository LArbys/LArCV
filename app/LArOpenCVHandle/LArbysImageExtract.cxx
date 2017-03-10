#ifndef __LARBYSIMAGEEXTRACT_CXX__
#define __LARBYSIMAGEEXTRACT_CXX__

#include "LArbysImageExtract.h"

namespace larcv {

  static LArbysImageExtractProcessFactory __global_LArbysImageExtractProcessFactory__;

  LArbysImageExtract::LArbysImageExtract(const std::string name)
    : ProcessBase(name),
      _LArbysImageMaker()
  {}
    
  void LArbysImageExtract::configure(const PSet& cfg)
  {
    _adc_producer = cfg.get<std::string>("ADCImageProducer");
    _track_producer = cfg.get<std::string>("TrackImageProducer");
    _shower_producer = cfg.get<std::string>("ShowerImageProducer");
    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
  }

  void LArbysImageExtract::initialize()
  {}

  bool LArbysImageExtract::process(IOManager& mgr)
  {

    _adc_mat_v = _LArbysImageMaker.ExtractMat(mgr,_adc_producer);
    _track_mat_v = _LArbysImageMaker.ExtractMat(mgr,_track_producer);
    _shower_mat_v = _LArbysImageMaker.ExtractMat(mgr,_shower_producer);

    return true;
  }

  void LArbysImageExtract::finalize()
  {}

}
#endif
