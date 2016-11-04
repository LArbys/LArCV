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
    _producer  = cfg.get<std::string>("MCProducer");
  }

  void MCinfoRetriever::initialize()
  {
    _mc_tree = new TTree("mctree","MC infomation");
    _mc_tree->Branch("run",&_run,"sun/I");
    _mc_tree->Branch("subrun",&_subrun,"subrun/I");
    _mc_tree->Branch("event",&_event,"event/I");
    _mc_tree->Branch("parentPDG",&_parent_pdg,"parentPDG/I");
    _mc_tree->Branch("energyDeposit",&_energy_deposit,"energyDeposit/D");
    _mc_tree->Branch("parentX",&_parent_x,"parentX/D");
    _mc_tree->Branch("parentY",&_parent_y,"parentY/D");
    _mc_tree->Branch("parentZ",&_parent_z,"parentZ/D");
    _mc_tree->Branch("parentT",&_parent_t,"parentT/D");
    _mc_tree->Branch("parentPx",&_parent_px,"parentPx/D");
    _mc_tree->Branch("parentPy",&_parent_py,"parentPy/D");
    _mc_tree->Branch("parentPz",&_parent_pz,"parentpz/D");
    _mc_tree->Branch("currentType",&_current_type,"currentType/I");
    _mc_tree->Branch("interactionType",&_current_type,"InteractionType/I");
    
  }

  bool MCinfoRetriever::process(IOManager& mgr)
  {

    // get the ROI data that has the MC information
    //https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/EventROI.h
    //https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/ROI.h
    //std::cout << "This event is: " << ev_roi->event() << std::endl;

    auto ev_roi = (larcv::EventROI*)mgr.get_data(kProductROI,_producer);
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
