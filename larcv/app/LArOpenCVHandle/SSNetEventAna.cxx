#ifndef __SSNETEVENTANA_CXX__
#define __SSNETEVENTANA_CXX__

#include "SSNetEventAna.h"

#include "DataFormat/EventImage2D.h"

#include <cassert>
#include <array>

namespace larcv {

  static SSNetEventAnaProcessFactory __global_SSNetEventAnaProcessFactory__;

  SSNetEventAna::SSNetEventAna(const std::string name)
    : ProcessBase(name), _tree(nullptr)
  {}
    
  void SSNetEventAna::configure(const PSet& cfg)
  {
    _ev_img2d_prod   = cfg.get<std::string>("EventImage2DProducer");
    _ev_score0_prod  = cfg.get<std::string>("Score0Image2DProducer");
    _ev_score1_prod  = cfg.get<std::string>("Score1Image2DProducer");
    _ev_score2_prod  = cfg.get<std::string>("Score2Image2DProducer");
    _ev_trk2d_prod   = cfg.get<std::string>("TrackImage2DProducer");
    _ev_shr2d_prod   = cfg.get<std::string>("ShowerImage2DProducer");
    _threshold       = cfg.get<float>("Threshold",10);   
  }
  
  void SSNetEventAna::initialize()
  {
    
    _tree = new TTree("SSNetEventAna","");

    _tree->Branch("run",&_run,"run/I");
    _tree->Branch("subrun",&_subrun,"subrun/I");
    _tree->Branch("event",&_event,"event/I");
    _tree->Branch("entry",&_entry,"entry/I");
    
    _tree->Branch("n_trk_pixel0",&_n_trk_pixel0,"_n_trk_pixel0/F");
    _tree->Branch("n_trk_pixel1",&_n_trk_pixel1,"_n_trk_pixel1/F");
    _tree->Branch("n_trk_pixel2",&_n_trk_pixel2,"_n_trk_pixel2/F");

    _tree->Branch("avg_trk_pixel_val0", &_avg_trk_pixel_val0, "avg_trk_pixel_val0/F");
    _tree->Branch("avg_trk_pixel_val1", &_avg_trk_pixel_val1, "avg_trk_pixel_val1/F");
    _tree->Branch("avg_trk_pixel_val2", &_avg_trk_pixel_val2, "avg_trk_pixel_val2/F");

    _tree->Branch("std_trk_pixel_val0", &_std_trk_pixel_val0, "std_trk_pixel_val0/F");
    _tree->Branch("std_trk_pixel_val1", &_std_trk_pixel_val1, "std_trk_pixel_val1/F");
    _tree->Branch("std_trk_pixel_val2", &_std_trk_pixel_val2, "std_trk_pixel_val2/F");

    _tree->Branch("tot_trk_pixel_val0", &_tot_trk_pixel_val0, "tot_trk_pixel_val0/F");
    _tree->Branch("tot_trk_pixel_val1", &_tot_trk_pixel_val1, "tot_trk_pixel_val1/F");
    _tree->Branch("tot_trk_pixel_val2", &_tot_trk_pixel_val2, "tot_trk_pixel_val2/F");

    _tree->Branch("avg_trk_score0", &_avg_trk_score0, "avg_trk_score0/F");
    _tree->Branch("avg_trk_score1", &_avg_trk_score1, "avg_trk_score1/F");
    _tree->Branch("avg_trk_score2", &_avg_trk_score2, "avg_trk_score2/F");

    _tree->Branch("std_trk_score0", &_std_trk_score0, "std_trk_score0/F");
    _tree->Branch("std_trk_score1", &_std_trk_score1, "std_trk_score1/F");
    _tree->Branch("std_trk_score2", &_std_trk_score2, "std_trk_score2/F");

    _tree->Branch("n_shr_pixel0",&_n_shr_pixel0,"n_shr_pixel0/F");
    _tree->Branch("n_shr_pixel1",&_n_shr_pixel1,"n_shr_pixel1/F");
    _tree->Branch("n_shr_pixel2",&_n_shr_pixel2,"n_shr_pixel2/F");

    _tree->Branch("avg_shr_pixel_val0", &_avg_shr_pixel_val0, "avg_shr_pixel_val0/F");
    _tree->Branch("avg_shr_pixel_val1", &_avg_shr_pixel_val1, "avg_shr_pixel_val1/F");
    _tree->Branch("avg_shr_pixel_val2", &_avg_shr_pixel_val2, "avg_shr_pixel_val2/F");

    _tree->Branch("std_shr_pixel_val0", &_std_shr_pixel_val0, "std_shr_pixel_val0/F");
    _tree->Branch("std_shr_pixel_val1", &_std_shr_pixel_val1, "std_shr_pixel_val1/F");
    _tree->Branch("std_shr_pixel_val2", &_std_shr_pixel_val2, "std_shr_pixel_val2/F");

    _tree->Branch("tot_shr_pixel_val0", &_tot_shr_pixel_val0, "tot_shr_pixel_val0/F");
    _tree->Branch("tot_shr_pixel_val1", &_tot_shr_pixel_val1, "tot_shr_pixel_val1/F");
    _tree->Branch("tot_shr_pixel_val2", &_tot_shr_pixel_val2, "tot_shr_pixel_val2/F");

    _tree->Branch("avg_shr_score0", &_avg_shr_score0, "avg_shr_score0/F");
    _tree->Branch("avg_shr_score1", &_avg_shr_score1, "avg_shr_score1/F");
    _tree->Branch("avg_shr_score2", &_avg_shr_score2, "avg_shr_score2/F");

    _tree->Branch("std_shr_score0", &_std_shr_score0, "std_shr_score0/F");
    _tree->Branch("std_shr_score1", &_std_shr_score1, "std_shr_score1/F");
    _tree->Branch("std_shr_score2", &_std_shr_score2, "std_shr_score2/F");

    _tree->Branch("n_adcpixel0",&_n_adc_pixel0,"n_adc_pixel0/F");
    _tree->Branch("n_adcpixel1",&_n_adc_pixel1,"n_adc_pixel1/F");
    _tree->Branch("n_adcpixel2",&_n_adc_pixel2,"n_adc_pixel2/F");

    _tree->Branch("avg_adc_pixel_val0", &_avg_adc_pixel_val0, "avg_adc_pixel_val0/F");
    _tree->Branch("avg_adc_pixel_val1", &_avg_adc_pixel_val1, "avg_adc_pixel_val1/F");
    _tree->Branch("avg_adc_pixel_val2", &_avg_adc_pixel_val2, "avg_adc_pixel_val2/F");

    _tree->Branch("std_adc_pixel_val0", &_std_adc_pixel_val0, "std_adc_pixel_val0/F");
    _tree->Branch("std_adc_pixel_val1", &_std_adc_pixel_val1, "std_adc_pixel_val1/F");
    _tree->Branch("std_adc_pixel_val2", &_std_adc_pixel_val2, "std_adc_pixel_val2/F");

    _tree->Branch("tot_adc_pixel_val0", &_tot_adc_pixel_val0, "tot_adc_pixel_val0/F");
    _tree->Branch("tot_adc_pixel_val1", &_tot_adc_pixel_val1, "tot_adc_pixel_val1/F");
    _tree->Branch("tot_adc_pixel_val2", &_tot_adc_pixel_val2, "tot_adc_pixel_val2/F");

    Reset();
  }

  bool SSNetEventAna::process(IOManager& mgr) 
  {
    Reset();

    const auto ev_img2d  = (EventImage2D*)(mgr.get_data(kProductImage2D,_ev_img2d_prod));
    if (!ev_img2d) throw larbys("Invalid event image producer provided");

    const auto ev_score0  = (EventImage2D*)(mgr.get_data(kProductImage2D,_ev_score0_prod));
    if (!ev_score0) throw larbys("Invalid event score producer provided");

    const auto ev_score1  = (EventImage2D*)(mgr.get_data(kProductImage2D,_ev_score1_prod));
    if (!ev_score1) throw larbys("Invalid event score producer provided");

    const auto ev_score2  = (EventImage2D*)(mgr.get_data(kProductImage2D,_ev_score2_prod));
    if (!ev_score2) throw larbys("Invalid event score producer provided");

    const auto ev_trk2d  = (EventImage2D*)(mgr.get_data(kProductImage2D,_ev_trk2d_prod));
    if (!ev_trk2d) throw larbys("Invalid event trk producer provided");

    const auto ev_shr2d  = (EventImage2D*)(mgr.get_data(kProductImage2D,_ev_shr2d_prod));
    if (!ev_shr2d) throw larbys("Invalid event shr producer provided");  

    _run    = (int) ev_img2d->run();
    _subrun = (int) ev_img2d->subrun();
    _event  = (int) ev_img2d->event();
    _entry  = (int) mgr.current_entry();

    std::array<float,3> n_trk_pixel_v;
    std::array<float,3> avg_trk_pixel_val_v;
    std::array<float,3> std_trk_pixel_val_v;
    std::array<float,3> tot_trk_pixel_val_v;
    std::array<float,3> avg_trk_score_v;
    std::array<float,3> std_trk_score_v;

    std::array<float,3> n_shr_pixel_v;
    std::array<float,3> avg_shr_pixel_val_v;
    std::array<float,3> std_shr_pixel_val_v;
    std::array<float,3> tot_shr_pixel_val_v;
    std::array<float,3> avg_shr_score_v;
    std::array<float,3> std_shr_score_v;

    std::array<float,3> n_adc_pixel_v;
    std::array<float,3> avg_adc_pixel_val_v;
    std::array<float,3> std_adc_pixel_val_v;
    std::array<float,3> tot_adc_pixel_val_v;

    for(size_t plane=0; plane<3; ++plane) {
      n_trk_pixel_v[plane] = 0;
      avg_trk_pixel_val_v[plane] = 0;
      std_trk_pixel_val_v[plane] = 0;
      tot_trk_pixel_val_v[plane] = 0;
      avg_trk_score_v[plane] = 0;
      std_trk_score_v[plane] = 0;

      n_shr_pixel_v[plane] = 0;
      avg_shr_pixel_val_v[plane] = 0;
      std_shr_pixel_val_v[plane] = 0;
      tot_shr_pixel_val_v[plane] = 0;
      avg_shr_score_v[plane] = 0;
      std_shr_score_v[plane] = 0;

      n_adc_pixel_v[plane] = 0;
      avg_adc_pixel_val_v[plane] = 0;
      std_adc_pixel_val_v[plane] = 0;
      tot_adc_pixel_val_v[plane] = 0;
    }
    
    for(size_t plane=0; plane<3; ++plane) {
      const auto& adc_img   = ev_img2d->Image2DArray()[plane];
      const auto& trk_img   = ev_trk2d->Image2DArray()[plane];
      const auto& shr_img   = ev_shr2d->Image2DArray()[plane];
      
      const auto& adc_vec   = adc_img.as_vector();
      const auto& trk_vec   = trk_img.as_vector();
      const auto& shr_vec   = shr_img.as_vector();

      const larcv::Image2D* trk_score_img_ptr = nullptr;
      const larcv::Image2D* shr_score_img_ptr = nullptr;

      switch(plane) {
      case 0:  
	trk_score_img_ptr = &(ev_score0->Image2DArray().back()); 
	shr_score_img_ptr = &(ev_score0->Image2DArray().front()); 
	break;
      case 1:  
	trk_score_img_ptr = &(ev_score1->Image2DArray().back());
        shr_score_img_ptr = &(ev_score1->Image2DArray().front());
	break;
      case 2:  
        trk_score_img_ptr = &(ev_score2->Image2DArray().back());
        shr_score_img_ptr = &(ev_score2->Image2DArray().front());
	break;
      default: 
	throw larbys("die");
      }

      const auto& trk_score_vec = trk_score_img_ptr->as_vector();
      const auto& shr_score_vec = shr_score_img_ptr->as_vector();

      auto& n_adc_pixel = n_adc_pixel_v[plane];
      auto& avg_adc_pixel_val = avg_adc_pixel_val_v[plane];
      auto& std_adc_pixel_val = std_adc_pixel_val_v[plane];
      auto& tot_adc_pixel_val = tot_adc_pixel_val_v[plane];
      
      auto& n_trk_pixel = n_trk_pixel_v[plane];
      auto& avg_trk_pixel_val = avg_trk_pixel_val_v[plane];
      auto& std_trk_pixel_val = std_trk_pixel_val_v[plane];
      auto& tot_trk_pixel_val = tot_trk_pixel_val_v[plane];

      auto& avg_trk_score = avg_trk_score_v[plane];
      auto& std_trk_score = std_trk_score_v[plane];

      auto& n_shr_pixel = n_shr_pixel_v[plane];
      auto& avg_shr_pixel_val = avg_shr_pixel_val_v[plane];
      auto& std_shr_pixel_val = std_shr_pixel_val_v[plane];
      auto& tot_shr_pixel_val = tot_shr_pixel_val_v[plane];

      auto& avg_shr_score = avg_shr_score_v[plane];
      auto& std_shr_score = std_shr_score_v[plane];


      //
      // avg
      //

      for(size_t ix=0; ix<adc_vec.size(); ++ix) {
	if (adc_vec[ix] < _threshold) continue;
	n_adc_pixel       += 1;
	avg_adc_pixel_val += adc_vec[ix];
	tot_adc_pixel_val += adc_vec[ix];

	if (trk_vec[ix] > _threshold) {
	  n_trk_pixel       += 1;
	  avg_trk_pixel_val += trk_vec[ix];
	  tot_trk_pixel_val += trk_vec[ix];
	}

	if (shr_vec[ix] > _threshold)  {
	  n_shr_pixel       += 1;
	  avg_shr_pixel_val += shr_vec[ix];
	  tot_shr_pixel_val += shr_vec[ix];
	}

      }

      assert (shr_score_vec.size() == trk_score_vec.size());

      size_t n_trk_score = 0;
      size_t n_shr_score = 0;
      for(size_t ix=0; ix < shr_score_vec.size(); ++ix) {
	if (trk_score_vec[ix]>0) {
	  avg_trk_score += trk_score_vec[ix];
	  n_trk_score += 1;
	}
	if(shr_score_vec[ix]>0) {
	  avg_shr_score += shr_score_vec[ix];
	  n_shr_score += 1;
	}
      }

      avg_adc_pixel_val /= n_adc_pixel;
      avg_trk_pixel_val /= n_trk_pixel;
      avg_shr_pixel_val /= n_shr_pixel;

      avg_trk_score /= n_trk_score;
      avg_shr_score /= n_shr_score;

      //
      // std
      //
      for(size_t ix=0; ix<adc_vec.size(); ++ix) {
	if (adc_vec[ix] < _threshold) continue;
	std_adc_pixel_val += std::pow(adc_vec[ix] - avg_adc_pixel_val,2);

	if (trk_vec[ix] > _threshold) {
	  std_trk_pixel_val += std::pow(trk_vec[ix] - avg_trk_pixel_val,2);
	}
	if (shr_vec[ix] > _threshold)  {
	  std_shr_pixel_val += std::pow(shr_vec[ix] - avg_shr_pixel_val,2);
	}

      }

      for(size_t ix=0; ix < shr_score_vec.size(); ++ix) {
	if (trk_score_vec[ix] > 0)
	  std_trk_score += std::pow(trk_score_vec[ix] - avg_trk_score,2);

	if (shr_score_vec[ix] > 0)
	  std_shr_score += std::pow(shr_score_vec[ix] - avg_shr_score,2);
      }
      

      std_adc_pixel_val /= (n_adc_pixel - 1);

      std_trk_pixel_val /= (n_trk_pixel - 1);
      std_shr_pixel_val /= (n_shr_pixel - 1);

      std_trk_score /= (n_trk_score - 1);
      std_shr_score /= (n_shr_score - 1);

      std_adc_pixel_val = std::sqrt(std_adc_pixel_val);

      std_trk_pixel_val = std::sqrt(std_trk_pixel_val);
      std_shr_pixel_val = std::sqrt(std_shr_pixel_val);
      
      std_trk_score = std::sqrt(std_trk_score);
      std_shr_score = std::sqrt(std_shr_score);
      
    } // end plane
    
    //
    // plane 0
    //
    _n_trk_pixel0   = n_trk_pixel_v[0];
    _avg_trk_pixel_val0 = avg_trk_pixel_val_v[0];
    _std_trk_pixel_val0 = std_trk_pixel_val_v[0];
    _tot_trk_pixel_val0 = tot_trk_pixel_val_v[0];

    _avg_trk_score0 = avg_trk_score_v[0];
    _std_trk_score0 = std_trk_score_v[0];

    _n_shr_pixel0   = n_shr_pixel_v[0];
    _avg_shr_pixel_val0 = avg_shr_pixel_val_v[0];
    _std_shr_pixel_val0 = std_shr_pixel_val_v[0];
    _tot_shr_pixel_val0 = tot_shr_pixel_val_v[0];

    _avg_shr_score0 = avg_shr_score_v[0];
    _std_shr_score0 = std_shr_score_v[0];

    _n_adc_pixel0   = n_adc_pixel_v[0];
    _avg_adc_pixel_val0 = avg_adc_pixel_val_v[0];
    _std_adc_pixel_val0 = std_adc_pixel_val_v[0];
    _tot_adc_pixel_val0 = tot_adc_pixel_val_v[0];
    
    //
    // plane 1
    //

    _n_trk_pixel1   = n_trk_pixel_v[1];
    _avg_trk_pixel_val1 = avg_trk_pixel_val_v[1];
    _std_trk_pixel_val1 = std_trk_pixel_val_v[1];
    _tot_trk_pixel_val1 = tot_trk_pixel_val_v[1];

    _avg_trk_score1 = avg_trk_score_v[1];
    _std_trk_score1 = std_trk_score_v[1];

    _n_shr_pixel1   = n_shr_pixel_v[1];
    _avg_shr_pixel_val1 = avg_shr_pixel_val_v[1];
    _std_shr_pixel_val1 = std_shr_pixel_val_v[1];
    _tot_shr_pixel_val1 = tot_shr_pixel_val_v[1];

    _avg_shr_score1 = avg_shr_score_v[1];
    _std_shr_score1 = std_shr_score_v[1];

    _n_adc_pixel1   = n_adc_pixel_v[1];
    _avg_adc_pixel_val1 = avg_adc_pixel_val_v[1];
    _std_adc_pixel_val1 = std_adc_pixel_val_v[1];
    _tot_adc_pixel_val1 = tot_adc_pixel_val_v[1];


    //
    // plane 2
    //

    _n_trk_pixel2   = n_trk_pixel_v[2];
    _avg_trk_pixel_val2 = avg_trk_pixel_val_v[2];
    _std_trk_pixel_val2 = std_trk_pixel_val_v[2];
    _tot_trk_pixel_val2 = tot_trk_pixel_val_v[2];

    _avg_trk_score2 = avg_trk_score_v[2];
    _std_trk_score2 = std_trk_score_v[2];

    _n_shr_pixel2   = n_shr_pixel_v[2];
    _avg_shr_pixel_val2 = avg_shr_pixel_val_v[2];
    _std_shr_pixel_val2 = std_shr_pixel_val_v[2];
    _tot_shr_pixel_val2 = tot_shr_pixel_val_v[2];

    _avg_shr_score2 = avg_shr_score_v[2];
    _std_shr_score2 = std_shr_score_v[2];

    _n_adc_pixel2   = n_adc_pixel_v[2];
    _avg_adc_pixel_val2 = avg_adc_pixel_val_v[2];
    _std_adc_pixel_val2 = std_adc_pixel_val_v[2];
    _tot_adc_pixel_val2 = tot_adc_pixel_val_v[2];

    _tree->Fill();

    return true;
  } // end process

  void SSNetEventAna::Reset() {

    _run = kINVALID_INT;
    _subrun = kINVALID_INT;
    _event = kINVALID_INT;
    _entry = kINVALID_INT;
    
    _n_trk_pixel0 = kINVALID_FLOAT;
    _n_trk_pixel1 = kINVALID_FLOAT;
    _n_trk_pixel2 = kINVALID_FLOAT;

    _avg_trk_pixel_val0 = kINVALID_FLOAT;
    _avg_trk_pixel_val1 = kINVALID_FLOAT;
    _avg_trk_pixel_val2 = kINVALID_FLOAT;

    _std_trk_pixel_val0 = kINVALID_FLOAT;
    _std_trk_pixel_val1 = kINVALID_FLOAT;
    _std_trk_pixel_val2 = kINVALID_FLOAT;

    _tot_trk_pixel_val0 = kINVALID_FLOAT;
    _tot_trk_pixel_val1 = kINVALID_FLOAT;
    _tot_trk_pixel_val2 = kINVALID_FLOAT;

    _avg_trk_score0 = kINVALID_FLOAT;
    _avg_trk_score1 = kINVALID_FLOAT;
    _avg_trk_score2 = kINVALID_FLOAT;

    _std_trk_score0 = kINVALID_FLOAT;
    _std_trk_score1 = kINVALID_FLOAT;
    _std_trk_score2 = kINVALID_FLOAT;

    _n_shr_pixel0 = kINVALID_FLOAT;
    _n_shr_pixel1 = kINVALID_FLOAT;
    _n_shr_pixel2 = kINVALID_FLOAT;

    _avg_shr_pixel_val0 = kINVALID_FLOAT;
    _avg_shr_pixel_val1 = kINVALID_FLOAT;
    _avg_shr_pixel_val2 = kINVALID_FLOAT;

    _std_shr_pixel_val0 = kINVALID_FLOAT;
    _std_shr_pixel_val1 = kINVALID_FLOAT;
    _std_shr_pixel_val2 = kINVALID_FLOAT;

    _tot_shr_pixel_val0 = kINVALID_FLOAT;
    _tot_shr_pixel_val1 = kINVALID_FLOAT;
    _tot_shr_pixel_val2 = kINVALID_FLOAT;

    _avg_shr_score0 = kINVALID_FLOAT;
    _avg_shr_score1 = kINVALID_FLOAT;
    _avg_shr_score2 = kINVALID_FLOAT;

    _std_shr_score0 = kINVALID_FLOAT;
    _std_shr_score1 = kINVALID_FLOAT;
    _std_shr_score2 = kINVALID_FLOAT;

    _n_adc_pixel0 = kINVALID_FLOAT;
    _n_adc_pixel1 = kINVALID_FLOAT;
    _n_adc_pixel2 = kINVALID_FLOAT;

    _avg_adc_pixel_val0 = kINVALID_FLOAT;
    _avg_adc_pixel_val1 = kINVALID_FLOAT;
    _avg_adc_pixel_val2 = kINVALID_FLOAT;

    _std_adc_pixel_val0 = kINVALID_FLOAT;
    _std_adc_pixel_val1 = kINVALID_FLOAT;
    _std_adc_pixel_val2 = kINVALID_FLOAT;

    _tot_adc_pixel_val0 = kINVALID_FLOAT;
    _tot_adc_pixel_val1 = kINVALID_FLOAT;
    _tot_adc_pixel_val2 = kINVALID_FLOAT;

  }

  void SSNetEventAna::finalize()
  {
    _tree->Write();
  
  }

}
#endif
