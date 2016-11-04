#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name),
      _reco_tree(nullptr)
  {}
    
  void LArbysImageAna::configure(const PSet& cfg)
  {
  }

  void LArbysImageAna::initialize()
  {
    _reco_tree = new TTree("LArbysImageTree","");
    _reco_tree->Branch("run"    ,&_run    , "run/i");
    _reco_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _reco_tree->Branch("event"  ,&_event  , "event/i");

    std::cout << "ImageClusterManager pointer is: " << _mgr_ptr << std::endl;
    
  }
  
  bool LArbysImageAna::process(IOManager& mgr)
  {

    
    /// Unique event keys
    const auto& event_id = mgr.event_id();
    _run    = (uint)event_id.run();
    _subrun = (uint)event_id.subrun();
    _event  = (uint)event_id.event();


    return true;
  }

  void LArbysImageAna::finalize()
  {
    _reco_tree->Write();
  }

}
#endif
