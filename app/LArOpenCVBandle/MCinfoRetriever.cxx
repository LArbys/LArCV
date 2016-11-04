#ifndef __MCINFORETRIEVER_CXX__
#define __MCINFORETRIEVER_CXX__

#include "MCinfoRetriever.h"
#include "DataFormat/ROI.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static MCinfoRetrieverProcessFactory __global_MCinfoRetrieverProcessFactory__;

  MCinfoRetriever::MCinfoRetriever(const std::string name)
    : ProcessBase(name)
  {}
    
  void MCinfoRetriever::configure(const PSet& cfg)
  {
    
  }

  void MCinfoRetriever::initialize()
  {
    _mc_tree = new TTree("mctree","MC infomation");
    _mc_tree->Branch("run",&_run,"sun/I");
    _mc_tree->Branch("subrun",&_subrun,"subrun/I");
    _mc_tree->Branch("event",&_event,"event/I");

  }

  bool MCinfoRetriever::process(IOManager& mgr)
  {

    // get the ROI data that has the MC information
    //https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/EventROI.h
    //https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/ROI.h
    
    auto ev_roi = (larcv::EventROI*)mgr.get_data(kProductROI,"tpc_hires_crop");
    
    //auto ev_roi = (larcv::EventROI*)(mgr.get_data(kProductROI,"tpc"));
    //std::cout << "This event is: " << ev_roi->event() << std::endl;
    _run    = ev_roi->run();
    _subrun = ev_roi->subrun();
    _run    = ev_roi->event();
    
    auto roi = ev_roi->at(0);
    
    _parent_pdg = roi.PdgCode();
    _energy_deposit = roi.EnergyDeposit();
    _parent_x = roi.X(); 
    _parent_y = roi.Y(); 
    _parent_z = roi.Z(); 
    _parent_t = roi.T(); 
    _parent_px = roi.Px(); 
    _parent_py = roi.Py(); 
    _parent_pz = roi.Pz(); 

    _current_type = roi.NuCurrentType();
    _interaction_type  =roi.NuInteractionType();
    
    _mc_tree->Fill();
    return true;
    
  }

  void MCinfoRetriever::finalize()
  {
    _mc_tree->Write();
  }

}
#endif
