#ifndef __NUFILTER_CXX__
#define __NUFILTER_CXX__

#include "NuFilter.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static NuFilterProcessFactory __global_NuFilterProcessFactory__;

  NuFilter::NuFilter(const std::string name)
    : ProcessBase(name)
  {}
    
  void NuFilter::configure(const PSet& cfg)
  {
    _nu_pdg         = cfg.get<unsigned int>("NuPDG");
    
    _min_nu_init_e  = cfg.get<double>("MinNuEnergy");
    _max_nu_init_e  = cfg.get<double>("MaxNuEnergy");
    _dep_sum_lepton = cfg.get<double>("MinEDepSumLepton");
    _dep_sum_proton = cfg.get<double>("MinEDepSumProton");
    
    _select_signal     = cfg.get<bool>("SelectSignal");
    _select_background = cfg.get<bool>("SelectBackground");

    _roi_producer_name = cfg.get<std::string>("ROIProducer");
  }

  //from rui an.
  bool NuFilter::MCSelect(const EventROI* ev_roi) {
    
    bool pdg_b            = false;
    bool engini_b         = false;
    bool nlepton_b        = false;
    bool dep_sum_lepton_b = false;
    bool nproton_b        = false;
    bool dep_sum_proton_b = false;
    //bool nprimary_b       = false;
    bool vis_one_proton_b = false;
    
    uint nlepton=0;
    uint nproton=0;
    uint nprimary=0;
    
    double dep_sum_lepton=0.0;
    double dep_sum_proton=0.0;
    
    // Get neutrino ROI
    auto roi = ev_roi->at(0);
    
    uint pdgcode = roi.PdgCode();
    
    //double energy_deposit = roi.EnergyDeposit();
    double energy_init    = roi.EnergyInit();

    std::vector<this_proton> protons_v;
    protons_v.clear();
    protons_v.reserve(10);

    uint ic = 0 ; 
    
    for(const auto& roi : ev_roi->ROIArray()) {

      if (ic==0)
	{ ic+=1; continue; }      
      
      int pdgcode = roi.PdgCode();
      
      if (pdgcode==2212) {


	if (roi.TrackID() == roi.ParentTrackID())
	  nproton++;
	
	//all protons go into vector
	this_proton thispro;
	
	thispro.trackid       = roi.TrackID();
	thispro.parenttrackid = roi.ParentTrackID();
	thispro.depeng        = roi.EnergyDeposit();
	
	protons_v.push_back(thispro);
      }

      //must be a parent particle
      if (roi.TrackID() != roi.ParentTrackID()) continue;

      nprimary+=1;
      
      if (pdgcode==11 or pdgcode==-11 or
	  pdgcode==13 or pdgcode==-13) {
	nlepton++;
	dep_sum_lepton += roi.EnergyDeposit();
      }
    }
    

    float highest_primary_proton_eng = 0;
    std::vector<float> proton_engs;
    proton_engs.clear();
    proton_engs.reserve(10);
      
    int proton_engs_ctr = 0 ;
    if (protons_v.size() > 0){
      int trackid ;
      int ptrackid ;
      for (int x=0;x < protons_v.size() ; x++ ){
        highest_primary_proton_eng = 0;
        trackid = protons_v.at(x).trackid;
        ptrackid = protons_v.at(x).parenttrackid;
	highest_primary_proton_eng += protons_v.at(x).depeng;

	if (highest_primary_proton_eng>_dep_sum_proton && trackid == ptrackid) proton_engs_ctr ++;
	
	for (int y=0;y < protons_v.size() ; y++ ){
          if (x==y) continue;
          if (protons_v.at(y).parenttrackid == trackid) {
	    highest_primary_proton_eng+=protons_v.at(y).depeng;
          }
        }
	
	proton_engs.push_back(highest_primary_proton_eng);
      }
      
      highest_primary_proton_eng = 0;
      for (auto const each : proton_engs) {
        if (each > highest_primary_proton_eng)
	  highest_primary_proton_eng = each;
      }
      dep_sum_proton = highest_primary_proton_eng;
    }

    if ( pdgcode == _nu_pdg)  pdg_b = true;
    if (energy_init >= _min_nu_init_e && energy_init <= _max_nu_init_e) engini_b = true;
    // Interactions could have more than one lepton.
    if (nlepton > 0) nlepton_b         = true; 
    if (dep_sum_lepton>_dep_sum_lepton) dep_sum_lepton_b  = true; // 
    // Interactions could have more than one proton(only one visible).
    if (nproton >= 1)  nproton_b         = true; 
    if (dep_sum_proton>=_dep_sum_proton) dep_sum_proton_b  = true; 
    // Note that cases where no vis proton is included. This should be thought as intrinsic error
    if (proton_engs_ctr <=1) vis_one_proton_b  = true;
    //not used
    //if (nprimary == 2)       nprimary_b        = true;

    bool selected = pdg_b && engini_b && nlepton_b && dep_sum_lepton_b && nproton_b && dep_sum_proton_b && vis_one_proton_b;
    
    LARCV_DEBUG() << "<<<<<<<<<<<<<<<<<<<<<<<<<"
	      << "selected is     " << selected        << "\n"
	      << "pdgcode is      " << pdgcode         << "\t~~~pdg:            " << pdg_b << "\n"
	      << "nu init e is    " << energy_init     << "\t~~engini_b:        " << engini_b << "\n"
	      << "nlepton is      " << nlepton        << "\t~~~nlepton:        " << nlepton_b <<'\n'
	      << "dep lepton is   " << dep_sum_lepton << "\t~~~deplepton:      " << dep_sum_lepton_b <<'\n'
	      << "nproton is      " << nproton       << "\t~~~nproton:        " << nproton_b <<'\n'
	      << "dep proton is   " << dep_sum_proton << "\t~~~deppro:         " << dep_sum_proton_b <<'\n'
	      << "proton_engs_ctr " << proton_engs_ctr  << "\t~~~vis_one_proton: " << vis_one_proton_b << std::endl;
      //<< "n primary is    " << nprimary       << "\t~~~nprimary:       " << nprimary_b <<'\n';

    _selected = selected;
    return selected;
  }

  void NuFilter::initialize()
  {}

  bool NuFilter::process(IOManager& mgr)
  {

    auto ev_roi = (EventROI*)mgr.get_data(kProductROI, _roi_producer_name);
    
    bool signal_selected = MCSelect(ev_roi);

    // if atleast 1 of config selection is false, then test against signal selected
    if ( !_select_signal or !_select_background) {
      if ( _select_signal     and !signal_selected ) return false;
      if ( _select_background and  signal_selected ) return false;
    }
    
    return true;
  }
  
  void NuFilter::finalize()
  {}

}
#endif
