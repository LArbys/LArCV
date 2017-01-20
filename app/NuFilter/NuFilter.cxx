#ifndef __NUFILTER_CXX__
#define __NUFILTER_CXX__

#include "NuFilter.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static NuFilterProcessFactory __global_NuFilterProcessFactory__;

  NuFilter::NuFilter(const std::string name)
    : ProcessBase(name), _event_tree(nullptr)
  {
    _n_calls = 0;
    _n_fail_nupdg = 0;
    _n_fail_nuE = 0;
    _n_fail_lepton_dep = 0;
    _n_fail_proton_dep = 0;
    _n_pass = 0;

    _event_tree = new TTree("event_tree","");
    _event_tree->Branch("run"   ,&_run,    "run/i");
    _event_tree->Branch("subrun",&_subrun, "subrun/i");
    _event_tree->Branch("event" ,&_event,  "event/i");
    
  }
    
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
    _n_calls+=1;
    
    bool pdg_b            = false;
    bool engini_b         = false;
    bool nlepton_b        = false;
    bool dep_sum_lepton_b = false;
    bool nproton_b        = false;
    bool dep_sum_proton_b = false;
    //bool nprimary_b       = false;
    bool vis_lepton_b = false;
    bool vis_one_proton_b = false;
    
    uint nlepton  = 0;
    uint nproton  = 0;
    uint nprimary = 0;
    
    double dep_sum_lepton=0.0;
    double dep_sum_proton=0.0;
    
    // Get neutrino ROI
    auto roi = ev_roi->at(0);
    
    uint pdgcode = roi.PdgCode();
    
    double energy_init    = roi.EnergyInit();

    std::vector<aparticle> protons_v;
    std::vector<aparticle> leptons_v;

    uint ic = 0 ; 
    
    for(const auto& roi : ev_roi->ROIArray()) {

      if (ic==0) { ic+=1; continue; }      
      
      int pdgcode = std::abs(roi.PdgCode());
      
      if (pdgcode==2212) {

	aparticle thispro;
	
	thispro.trackid       = roi.TrackID();
	thispro.ptrackid      = roi.ParentTrackID();
	thispro.depeng        = roi.EnergyDeposit();
	thispro.primary       = ( roi.TrackID() == roi.ParentTrackID() );
	protons_v.push_back(std::move(thispro));
      }
      
      if (pdgcode==11 or pdgcode==13) {

	aparticle thislep;
	
	thislep.trackid       = roi.TrackID();
	thislep.ptrackid      = roi.ParentTrackID();
	thislep.depeng        = roi.EnergyDeposit();
	thislep.primary       = ( roi.TrackID() == roi.ParentTrackID() );
	leptons_v.push_back(std::move(thislep));
      }
      
    }
    
    //calculate the visible lepton energy
    int lepton_engs_ctr = 0 ;
    for (int p1=0;p1 < leptons_v.size() ; p1++ ){
      const auto& lepton1 = leptons_v[p1];
      if (! lepton1.primary ) continue;
      nlepton+=1;
      float this_lepton_eng = lepton1.depeng;
      for (int p2=0;p2 < leptons_v.size() ; p2++ ){
      	if (p1==p2) continue;
      	const auto& lepton2 = leptons_v[p2];
      	if (lepton2.ptrackid != lepton1.trackid) continue;
      	this_lepton_eng+=lepton2.depeng;
      }
      if ( this_lepton_eng > _dep_sum_lepton ) lepton_engs_ctr ++;
    }

    //calculate the visible proton energy
    int proton_engs_ctr = 0 ;
    for (int p1=0;p1 < protons_v.size() ; p1++ ){
      const auto& proton1 = protons_v[p1];
      if (! proton1.primary ) continue;
      nproton+=1;
      float this_proton_eng = proton1.depeng;
      for (int p2=0;p2 < protons_v.size() ; p2++ ){
	if (p1==p2) continue;
	const auto& proton2 = protons_v[p2];
	if (proton2.ptrackid != proton1.trackid) continue;
	this_proton_eng+=proton2.depeng;
      }
      if ( this_proton_eng > _dep_sum_proton ) proton_engs_ctr ++;
    }

    // requested nu PDG code
    if (pdgcode == _nu_pdg)  pdg_b=true;

    // neutrino energy range
    if (energy_init >= _min_nu_init_e && energy_init <= _max_nu_init_e) engini_b=true;

    // interaction must have visible leptons
    if (lepton_engs_ctr == 1) vis_lepton_b =true;

    // There must be 1 visible proton
    if (proton_engs_ctr == 1) vis_one_proton_b =true;

    bool selected = pdg_b && engini_b  && vis_lepton_b && vis_one_proton_b;
    
    LARCV_DEBUG() << "<<<<<<<<<<<<<<<<<<<<<<<<<"
		  << "selected is     " << selected        << "\n"
		  << "pdgcode is      " << pdgcode         << "\t~~~pdg:            " << pdg_b << "\n"
		  << "nu init e is    " << energy_init     << "\t~~engini_b:        " << engini_b << "\n"
		  << "nlepton is      " << nlepton          << "\t~~~nlepton:        " << nlepton_b <<'\n'
		  << "dep lepton is   " << dep_sum_lepton   << "\t~~~deplepton:      " << dep_sum_lepton_b <<'\n'
		  << "nproton is      " << nproton          << "\t~~~nproton:        " << nproton_b <<'\n'
		  << "dep proton is   " << dep_sum_proton   << "\t~~~deppro:         " << dep_sum_proton_b <<'\n'
		  << "proton_engs_ctr " << proton_engs_ctr  << "\t~~~vis_one_proton: " << vis_one_proton_b << std::endl;

    _selected = selected;


    if (! pdg_b)            _n_fail_nupdg      += 1;
    if (! engini_b)         _n_fail_nuE        += 1;
    if (! vis_lepton_b)     _n_fail_lepton_dep += 1;
    if (! vis_one_proton_b) _n_fail_proton_dep += 1;
    
    
    return selected;
  }

  void NuFilter::initialize()
  {}

  bool NuFilter::process(IOManager& mgr)
  {

    auto ev_roi = (EventROI*) mgr.get_data(kProductROI, _roi_producer_name);

    _run   = ev_roi->run();
    _subrun= ev_roi->subrun();
    _event = ev_roi->event();
    
    bool signal_selected = MCSelect(ev_roi);
    
    // if atleast 1 of config selection is false, then test against signal selected
    if ( !_select_signal or !_select_background) {
      if ( _select_signal     and !signal_selected ) return false;
      if ( _select_background and  signal_selected ) return false;
    }

    
    _n_pass+=1;

    _event_tree->Fill();
    
    return true;
  }
  
  void NuFilter::finalize()
  {
    std::cout << std::endl;
    std::cout << "\t<~~~~~ NuFilter configuration ~~~~~>" << std::endl;
    std::cout << "NuPDG: " << _nu_pdg << std::endl;
    std::cout << "MinNuEnergy: " << _min_nu_init_e << std::endl;
    std::cout << "MaxNuEnergy: " << _max_nu_init_e << std::endl;
    std::cout << "MinEDepSumLepton: " << _dep_sum_lepton<< std::endl;
    std::cout << "MinEDepSumProton: " << _dep_sum_proton<< std::endl;
    std::cout << "SelectSignal: " << _select_signal << std::endl;
    std::cout << "SelectBackground: " << _select_background << std::endl;
    std::cout << "ROIProducer: " << _roi_producer_name << std::endl;
    std::cout << std::endl;
    std::cout << "\t<~~~~~ NuFilter statistics ~~~~~>" << std::endl;
    std::cout << "Called: " << _n_calls << std::endl;
    std::cout << "N fail Nu PDG: " << _n_fail_nupdg << std::endl;
    std::cout << "N fail Nu E: " << _n_fail_nuE << std::endl;
    std::cout << "N fail lepton dep: " << _n_fail_lepton_dep << std::endl;
    std::cout << "N fail proton dep: " << _n_fail_proton_dep << std::endl;
    std::cout << "N pass : " << _n_pass << std::endl;
    std::cout << std::endl;

    _event_tree->Write();
    
  }

}
#endif
