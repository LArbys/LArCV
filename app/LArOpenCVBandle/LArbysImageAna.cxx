#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"
#include "AlgoData/HIPClusterData.h"
#include "AlgoData/Refine2DVertexData.h"
#include "AlgoData/VertexClusterData.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"

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
    _vertexcluster_name  = cfg.get<std::string>("VertexTrackClusterAlgoName");
  }

  void LArbysImageAna::Clear() {
    _n_mip_ctors_v.clear();
    _n_hip_ctors_v.clear();
    _n_mip_ctors_v.resize(3);
    _n_hip_ctors_v.resize(3);

    _vtx3d_n_planes_v.clear();

    _vtx3d_x_v.clear();
    _vtx3d_y_v.clear();
    _vtx3d_z_v.clear();

    _vtx2d_x_vv.clear();
    _vtx2d_y_vv.clear();
    
    _x_vv.clear();
    _y_vv.clear();

    _num_planes_v.clear();
    _num_clusters_vv.clear();
    _num_pixels_vv.clear();
    _num_pixel_frac_vv.clear();
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
    
    _reco_tree->Branch("vtx3d_x_v", &_vtx3d_x_v );
    _reco_tree->Branch("vtx3d_y_v", &_vtx3d_y_v );
    _reco_tree->Branch("vtx3d_z_v", &_vtx3d_z_v );

    _reco_tree->Branch("vtx2d_x_vv", &_vtx2d_x_vv );
    _reco_tree->Branch("vtx2d_y_vv", &_vtx2d_y_vv );

    _reco_tree->Branch("n_circle_vtx", &_n_circle_vtx, "n_circle_vtx/i" );

    _reco_tree->Branch("circle_vtx_x_vv",&_x_vv);
    _reco_tree->Branch("circle_vtx_y_vv",&_y_vv);
    
    /// VertexTrackCluster
    _reco_tree->Branch("n_vtx_cluster", &_n_vtx_cluster, "n_vtx_cluster/i");    

    _reco_tree->Branch("num_planes_v"      ,&_num_planes_v);
    _reco_tree->Branch("num_clusters_vv"   ,&_num_clusters_vv);
    _reco_tree->Branch("num_pixels_vv"     ,&_num_pixels_vv);
    _reco_tree->Branch("num_pixel_frac_vv" ,&_num_pixel_frac_vv);
    

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
    _vtx3d_x_v.resize(_n_vtx3d);
    _vtx3d_y_v.resize(_n_vtx3d);
    _vtx3d_z_v.resize(_n_vtx3d);
    
    _vtx3d_n_planes_v.resize(_n_vtx3d);
    
    for(uint vtx_id=0;vtx_id<_n_vtx3d;++vtx_id) {
      const auto& vtx3d = vtx3d_v[vtx_id];
      _vtx3d_n_planes_v[vtx_id] = vtx3d.num_planes;

      _vtx3d_x_v[vtx_id] = vtx3d.x;
      _vtx3d_y_v[vtx_id] = vtx3d.y;
      _vtx3d_z_v[vtx_id] = vtx3d.z;

      _vtx2d_x_vv.resize(3);
      _vtx2d_y_vv.resize(3);
      
      for(uint plane_id=0;plane_id<3;++plane_id) {
	auto& vtx2d_x_v = _vtx2d_x_vv[plane_id];
	auto& vtx2d_y_v = _vtx2d_y_vv[plane_id];

	vtx2d_x_v.push_back(vtx3d.vtx2d_v[plane_id].pt.x);
	vtx2d_y_v.push_back(vtx3d.vtx2d_v[plane_id].pt.y);
      }
    }

    // vec of circle vertex per 3d vtx id NOTE: VIC, HEY -- MAY NOT BE 1-to-1 w/ vertex3d I don't know yet
    const auto& circle_vtx_vv = refine2d_data->get_circle_vertex();
    uint circle_vtx_vv_size=circle_vtx_vv.size();
    _n_circle_vtx = circle_vtx_vv_size;
    
    _x_vv.resize(circle_vtx_vv_size);
    _y_vv.resize(circle_vtx_vv_size);
    
    for(uint i=0;i<circle_vtx_vv_size;++i) {
      const auto& circle_vtx_v = circle_vtx_vv[i];

      auto& _x_v = _x_vv[i];
      auto& _y_v = _y_vv[i];
      
      _x_v.resize(circle_vtx_v.size());
      _y_v.resize(circle_vtx_v.size());
      
      for(uint plane_id=0;plane_id<circle_vtx_v.size();++plane_id) {
	const auto& circle_vtx = circle_vtx_v[plane_id];
	const auto& circle_vtx_c = circle_vtx.center;
	_x_v[plane_id] = circle_vtx_c.x;
	_y_v[plane_id] = circle_vtx_c.y;
      }
      
    }
    
    // /// VertexCluster data
    const auto vtxtrkcluster_data = (larocv::data::VertexClusterArray*)dm.Data( dm.ID(_vertexcluster_name) );
    auto& vtx_cluster_v=  vtxtrkcluster_data->_vtx_cluster_v;
    _n_vtx_cluster = (uint) vtx_cluster_v.size();

    _num_planes_v.resize(_n_vtx_cluster);

    
    _num_clusters_vv.resize(_n_vtx_cluster);
    _num_pixels_vv.resize(_n_vtx_cluster);
    _num_pixel_frac_vv.resize(_n_vtx_cluster);
    
    for(uint px=0; px<_n_vtx_cluster; ++px) { 

      const auto& vtx_cluster = vtx_cluster_v[px];
      _num_planes_v[px]   = vtx_cluster.num_planes();

      auto& num_clusters_v   = _num_clusters_vv[px];
      auto& num_pixels_v     = _num_pixels_vv[px];
      auto& num_pixel_frac_v = _num_pixel_frac_vv[px];

      const auto n_planes = vtx_cluster.num_planes();
      
      num_clusters_v.resize(n_planes);
      num_pixels_v.resize(n_planes);
      num_pixel_frac_v.resize(n_planes);
      
      for(uint plane_id=0;plane_id<n_planes;++plane_id) {
	num_clusters_v[plane_id]  = vtx_cluster.num_clusters(plane_id);
	num_pixels_v[plane_id]    = vtx_cluster.num_pixels(plane_id);
	num_pixel_frac_v[plane_id]= vtx_cluster.num_pixel_fraction(plane_id);
      }
      
    }

      
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
