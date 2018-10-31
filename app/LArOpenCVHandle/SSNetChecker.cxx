#ifndef __SSNETCHECKER_CXX__
#define __SSNETCHECKER_CXX__

#include "SSNetChecker.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static SSNetCheckerProcessFactory __global_SSNetCheckerProcessFactory__;

  SSNetChecker::SSNetChecker(const std::string name)
    : ProcessBase(name), _outtree(nullptr)
  {}
    
  void SSNetChecker::configure(const PSet& cfg)
  {
    _rse_producer   = cfg.get<std::string>("RSEProducer");
    _adc_producer   = cfg.get<std::string>("ADCImageProducer");
    _ssnet_producer = cfg.get<std::string>("SSNetImageProducer");
    _roi_producer   = cfg.get<std::string>("ROIProducer");
  }

  void SSNetChecker::initialize()
  {
    _outtree = new TTree("SSNetChecker","");
    _outtree->Branch("run"      , &_run      , "run/I");
    _outtree->Branch("subrun"   , &_subrun   , "subrun/I");
    _outtree->Branch("event"    , &_event    , "event/I");
    _outtree->Branch("entry"    , &_entry    , "entry/I");
    _outtree->Branch("broken"   , &_broken   , "broken/I");
    _outtree->Branch("valid_roi", &_valid_roi, "valid_roi/I");
    _outtree->Branch("fname"    , &_fname);
  }

  bool SSNetChecker::process(IOManager& mgr)
  {
    auto rse_prod = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,_rse_producer);
    auto ev_adc   = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,_adc_producer);
    auto ev_ssnet = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,_ssnet_producer);
    auto ev_roi   = (larcv::EventROI*)mgr.get_data(larcv::kProductROI,_roi_producer);

    _run    = (int)rse_prod->run();
    _subrun = (int)rse_prod->subrun();
    _event  = (int)rse_prod->event();
    _entry  = (int)mgr.current_entry();
    _broken = 0;
    _valid_roi = 1;
    
    if (ev_adc->Image2DArray().size() != ev_ssnet->Image2DArray().size()) {
      throw larbys("ADC and SSNet image differ in n planes");
    }

    const auto& roi     = (*ev_roi).ROIArray().front();
    
    for(size_t plane=0; plane<3; ++plane) {
      const auto& adc_img = (*ev_adc).Image2DArray()[plane];
      const auto& adc     = adc_img.as_vector();
      const auto& ssnet   = (*ev_ssnet).Image2DArray()[plane].as_vector();
      if (adc_img.meta() == roi.BB(plane)) {
	LARCV_CRITICAL() << "Invalid ROI" << std::endl;
	  LARCV_CRITICAL() << "@(r,s,e,e)"
			   << "(" << _run
			   << "," << _subrun
			   << "," << _event
			   << "," << _entry << ")" << std::endl;
	_valid_roi = 0;
      }

      for(size_t px=0; px<adc.size(); ++px) {
	if (adc[px] != ssnet[px]) {
	  LARCV_CRITICAL() << "ADC image and SSNet Image Differ" << std::endl;
	  LARCV_CRITICAL() << "@(r,s,e,e)"
			   << "(" << _run
			   << "," << _subrun
			   << "," << _event
			   << "," << _entry << ")" << std::endl;
	  _broken = 1;
	  _outtree->Fill();
	  return true;
	}
      }
      
    }
    _outtree->Fill();
    return true;
  }

  void SSNetChecker::finalize()
  {
    _outtree->Write();
  }

}
#endif
