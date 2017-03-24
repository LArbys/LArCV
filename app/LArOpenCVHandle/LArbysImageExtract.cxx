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
    _thrumu_producer = cfg.get<std::string>("ThruMuProducer","");
    _stopmu_producer = cfg.get<std::string>("StopMuProducer","");
    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
    _PreProcessor.Configure(cfg.get<larcv::PSet>("PreProcessor"));
  }

  void LArbysImageExtract::initialize()
  {}

  bool LArbysImageExtract::process(IOManager& mgr)
  {

    const auto ev_adc = (EventImage2D*)mgr.get_data(kProductImage2D,_adc_producer);
    const auto ev_trk = (EventImage2D*)mgr.get_data(kProductImage2D,_track_producer);
    const auto ev_shr = (EventImage2D*)mgr.get_data(kProductImage2D,_shower_producer);
    EventPixel2D* ev_thrumu_pix = nullptr;
    EventPixel2D* ev_stopmu_pix = nullptr;

    if (!_thrumu_producer.empty())
      ev_thrumu_pix = (EventPixel2D*)mgr.get_data(kProductPixel2D,_thrumu_producer);
    if (!_stopmu_producer.empty())
      ev_stopmu_pix = (EventPixel2D*)mgr.get_data(kProductPixel2D,_stopmu_producer);

    _ev_adc = *ev_adc;
    _ev_trk = *ev_trk;
    _ev_shr = *ev_shr;
    
    _ev_thrumu_pix = *ev_thrumu_pix;
    _ev_stopmu_pix = *ev_stopmu_pix;
    
    auto adc_img_data_v = _LArbysImageMaker.ExtractImage(ev_adc->Image2DArray());
    _adc_mat_v.clear();
    _adc_mat_v.reserve(3);
    
    _adc_meta_v.clear();
    _adc_meta_v.reserve(3);
    
    for(auto& img_data : adc_img_data_v) {
      _adc_mat_v.emplace_back(std::move(std::get<0>(img_data)));
      _adc_meta_v.emplace_back(std::move(std::get<1>(img_data)));
    }
    
    _track_mat_v = _LArbysImageMaker.ExtractMat(ev_trk->Image2DArray());
    _shower_mat_v = _LArbysImageMaker.ExtractMat(ev_shr->Image2DArray());

    return true;
  }
  
  void LArbysImageExtract::finalize()
  {}

}
#endif
