#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"
#include "AlgoData/HIPClusterData.h"
#include "AlgoData/Refine2DVertexData.h"
#include "AlgoData/VertexClusterData.h"

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
    //_vertexcluster_name = cfg.get<std::string>("VertexTrackClusterAlgoName");
  }

  void LArbysImageAna::Clear() {
    _n_mip_ctors_v.clear();
    _n_hip_ctors_v.clear();
    _n_mip_ctors_v.resize(3);
    _n_hip_ctors_v.resize(3);
    _vtx3d_n_planes_v.clear();
    _x_v.clear();
    _y_v.clear();
    _z_v.clear();
    _x0_v.clear();
    _x1_v.clear();
    _x2_v.clear();
    _y0_v.clear();
    _y1_v.clear();
    _y2_v.clear();	
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

    _reco_tree->Branch("vtx3d_num_planes_v", &_vtx3d_n_planes_v);
    
    _reco_tree->Branch("vtx3d_x_v", &_x_v );
    _reco_tree->Branch("vtx3d_y_v", &_y_v );
    _reco_tree->Branch("vtx3d_z_v", &_z_v );

    _reco_tree->Branch("n_circle_vtx", &_n_circle_vtx, "n_circle_vtx/i" );

    _reco_tree->Branch("circle_vtx_x0_v",&_x0_v);
    _reco_tree->Branch("circle_vtx_x1_v",&_x1_v);
    _reco_tree->Branch("circle_vtx_x2_v",&_x2_v);

    _reco_tree->Branch("circle_vtx_y0_v",&_y0_v);
    _reco_tree->Branch("circle_vtx_y1_v",&_y1_v);
    _reco_tree->Branch("circle_vtx_y2_v",&_y2_v);
	
    /// VertexTrackCluster
    
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

    // vec of circle vertex per 3d vtx id
    _x_v.resize(_n_vtx3d);
    _y_v.resize(_n_vtx3d);
    _z_v.resize(_n_vtx3d);
    _vtx3d_n_planes_v.resize(_n_vtx3d);
    
    for(uint vtx_id=0;vtx_id<_n_vtx3d;++vtx_id) {
      const auto& vtx3d = vtx3d_v[vtx_id];
      _vtx3d_n_planes_v[vtx_id] = vtx3d.num_planes;
      _x_v[vtx_id] = vtx3d.x;
      _y_v[vtx_id] = vtx3d.y;
      _z_v[vtx_id] = vtx3d.z;
    }

    // vec of circle vertex per 3d vtx id NOTE: VIC, HEY -- MAY NOT BE 1-to-1 w/ vertex3d I don't know yet
    const auto& circle_vtx_vv = refine2d_data->get_circle_vertex();
    uint circle_vtx_vv_size=circle_vtx_vv.size();
    _n_circle_vtx = circle_vtx_vv_size;
    
    _x0_v.resize(circle_vtx_vv_size);
    //std::fill(v.begin(), v.end(), -1.0)
    _x1_v.resize(circle_vtx_vv_size);
    _x2_v.resize(circle_vtx_vv_size);

    _y0_v.resize(circle_vtx_vv_size);
    _y1_v.resize(circle_vtx_vv_size);
    _y2_v.resize(circle_vtx_vv_size);
    
    for(uint i=0;i<circle_vtx_vv_size;++i) {
      const auto& circle_vtx_v = circle_vtx_vv[i];
      for(uint plane_id=0;plane_id<circle_vtx_v.size();++plane_id) {
	const auto& circle_vtx = circle_vtx_v[plane_id];
	const auto& circle_vtx_c = circle_vtx.center;
	switch (plane_id) {
	case 0 : _x0_v.at(i) = circle_vtx_c.x; _y0_v.at(i) = circle_vtx_c.y; break;
	case 1 : _x1_v.at(i) = circle_vtx_c.x; _y1_v.at(i) = circle_vtx_c.y; break;
	case 2 : _x2_v.at(i) = circle_vtx_c.x; _y2_v.at(i) = circle_vtx_c.y; break;
	default: throw larbys("Invalid plane requested in the circle vertex loop");
	}
      }
    }
    
    // /// VertexCluster data
    // const auto vtxtrkcluster_data = (larocv::data::VertexClusterArray*)dm.Data( dm.ID(_vertexcluster_name) );
    
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
