#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"
#include "AlgoData/HIPClusterData.h"
#include "AlgoData/Refine2DVertexData.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name),
      _reco_tree(nullptr)
  {}
    
  void LArbysImageAna::configure(const PSet& cfg)
  {
    _hipcluster_name     = cfg.get<std::string>("HIPClusterAlgoName");
    _defectcluster_name  = cfg.get<std::string>("DefectClusterAlgoName");
    _pcacandidates_name  = cfg.get<std::string>("PCACandidatesAlgoName");
    _refine2dvertex_name = cfg.get<std::string>("Refine2DVertexAlgoName");
    
  }

  void LArbysImageAna::Clear() {
    _n_mip_ctors_v.clear();
    _n_hip_ctors_v.clear();
    _n_mip_ctors_v.resize(3);
    _n_hip_ctors_v.resize(3);
  }
  
  void LArbysImageAna::initialize()
  {

    Clear();
    
    _reco_tree = new TTree("LArbysImageTree","");

    /// Unique event keys
    _reco_tree->Branch("run"    ,&_run    , "run/i");
    _reco_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _reco_tree->Branch("event"  ,&_event  , "event/i");

    /// HIP cluster data
    _reco_tree->Branch("n_mip_ctors_v", &_n_mip_ctors_v );
    _reco_tree->Branch("n_hip_ctors_v", &_n_hip_ctors_v );
    
    /// Refine2D data
    _reco_tree->Branch("n_vtx3d", &_n_vtx3d, "n_vtx3d/i");
    
  }
  
  bool LArbysImageAna::process(IOManager& mgr)
  {

    LARCV_DEBUG() << "process" << std::endl;
    
    /// Unique event keys
    const auto& event_id = mgr.event_id();
    _run    = (uint)event_id.run();
    _subrun = (uint)event_id.subrun();
    _event  = (uint)event_id.event();

    const auto& dm  = _mgr_ptr->DataManager();    
    
    /// HIP cluster data
    const auto hipctor_data = (larocv::data::HIPClusterData*)dm.Data( dm.ID(_hipcluster_name) );
    
    for(uint plane_id=0;plane_id<3;++plane_id) {
      const auto& hipctor_plane_data = hipctor_data->_plane_data_v[plane_id];
      _n_mip_ctors_v[plane_id] = hipctor_plane_data.num_mip();
      _n_hip_ctors_v[plane_id] = hipctor_plane_data.num_hip();
    }

    /// Refine2D data
    const auto refine2d_data = (larocv::data::Refine2DVertexData*)dm.Data( dm.ID(_refine2dvertex_name) );

    const auto& vtx3d_v = refine2d_data->get_vertex();
    _n_vtx3d = (uint) vtx3d_v.size();
      
    _reco_tree->Fill();
    
    Clear();
    return true;
  }

  void LArbysImageAna::finalize()
  {
    _reco_tree->Write();
  }

}
#endif
