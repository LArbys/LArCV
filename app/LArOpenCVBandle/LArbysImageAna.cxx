#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"
#include "AlgoData/HIPClusterData.h"
#include "AlgoData/Refine2DVertexData.h"
#include "AlgoData/VertexClusterData.h"
#include "AlgoData/LinearVtxFilterData.h"

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
    _hipcluster_name       = cfg.get<std::string>("HIPClusterAlgoName");
    _defectcluster_name    = cfg.get<std::string>("DefectClusterAlgoName");
    _pcacandidates_name    = cfg.get<std::string>("PCACandidatesAlgoName");
    _refine2dvertex_name   = cfg.get<std::string>("Refine2DVertexAlgoName");
    _vertexcluster_name    = cfg.get<std::string>("VertexTrackClusterAlgoName");
    _linearvtxfilter_name  = cfg.get<std::string>("LinearVtxFilterAlgoName");
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
    
    _circle_x_vv.clear();
    _circle_y_vv.clear();

    _num_planes_v.clear();
    _num_clusters_vv.clear();
    _num_pixels_vv.clear();
    _num_pixel_frac_vv.clear();
    
    _circle_vtx_r_vv.clear();
    _circle_vtx_angle_vv.clear();
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
    
    /// VertexTrackCluster
    _reco_tree->Branch("n_vtx3d", &_n_vtx3d, "n_vtx3d/i");

    _reco_tree->Branch("vtx3d_x_v", &_vtx3d_x_v );
    _reco_tree->Branch("vtx3d_y_v", &_vtx3d_y_v );
    _reco_tree->Branch("vtx3d_z_v", &_vtx3d_z_v );

    _reco_tree->Branch("vtx2d_x_vv", &_vtx2d_x_vv );
    _reco_tree->Branch("vtx2d_y_vv", &_vtx2d_y_vv );

    _reco_tree->Branch("circle_vtx_x_vv",&_circle_x_vv);
    _reco_tree->Branch("circle_vtx_y_vv",&_circle_y_vv);
    
    _reco_tree->Branch("num_planes_v"      ,&_num_planes_v);
    _reco_tree->Branch("num_clusters_vv"   ,&_num_clusters_vv);
    _reco_tree->Branch("num_pixels_vv"     ,&_num_pixels_vv);
    _reco_tree->Branch("num_pixel_frac_vv" ,&_num_pixel_frac_vv);


    //LinearVtxFilter
    _reco_tree->Branch("circle_vtx_r_vv",&_circle_vtx_r_vv);
    _reco_tree->Branch("circle_vtx_angle_vv",&_circle_vtx_angle_vv);
    
  }
  
  bool LArbysImageAna::process(IOManager& mgr)
  {
    
    LARCV_DEBUG() << "process" << std::endl;

    /// Unique event keys
    const auto& event_id = mgr.event_id();
    _run    = (uint) event_id.run();
    _subrun = (uint) event_id.subrun();
    _event  = (uint) event_id.event();

    const auto& dm  = _mgr_ptr->DataManager();    
    
    /// HIP cluster data
    const auto hipctor_data = (larocv::data::HIPClusterData*)dm.Data( dm.ID(_hipcluster_name) );
    
    for(uint plane_id=0;plane_id<3;++plane_id) {
      const auto& hipctor_plane_data = hipctor_data->_plane_data_v[plane_id];
      _n_mip_ctors_v[plane_id] = hipctor_plane_data.num_mip();
      _n_hip_ctors_v[plane_id] = hipctor_plane_data.num_hip();
    }

    /// Refine2D data
    /// const auto refine2d_data = (larocv::data::Refine2DVertexData*)dm.Data( dm.ID(_refine2dvertex_name) );

    /// VertexCluster data
    const auto vtxtrkcluster_data = (larocv::data::VertexClusterArray*)dm.Data( dm.ID(_vertexcluster_name) );


    /// LinearVtxFilter data
    auto linearvf_data = (larocv::data::LinearVtxFilterData*)dm.Data( dm.ID(_linearvtxfilter_name) );

    //careful: this is nonconst
    auto& circle_setting_array_v = linearvf_data->_circle_setting_array_v;
    auto& vtx_cluster_v=  vtxtrkcluster_data->_vtx_cluster_v;
    
    _n_vtx3d = (uint) vtx_cluster_v.size();

    // vec of circle vertex per 3d vtx id
    _vtx3d_x_v.resize(_n_vtx3d);
    _vtx3d_y_v.resize(_n_vtx3d);
    _vtx3d_z_v.resize(_n_vtx3d);

    _num_clusters_vv.resize(_n_vtx3d);
    _num_pixels_vv.resize(_n_vtx3d);
    _num_pixel_frac_vv.resize(_n_vtx3d);

    _vtx2d_x_vv.resize(_n_vtx3d);
    _vtx2d_y_vv.resize(_n_vtx3d);
    
    _circle_x_vv.resize(_n_vtx3d);
    _circle_y_vv.resize(_n_vtx3d);

    _num_planes_v.resize(_n_vtx3d);
    
    _circle_vtx_angle_vv.resize(_n_vtx3d);
    _circle_vtx_r_vv.resize(_n_vtx3d);
    
    for(uint vtx_id=0;vtx_id<_n_vtx3d;++vtx_id) {
      
      const auto& vtx_cluster = vtx_cluster_v[vtx_id];
      
      const auto& vtx3d = vtx_cluster.get_vertex();

      auto& csarray = circle_setting_array_v[vtx_id];
      
      _vtx3d_x_v[vtx_id] = vtx3d.x;
      _vtx3d_y_v[vtx_id] = vtx3d.y;
      _vtx3d_z_v[vtx_id] = vtx3d.z;

      _num_planes_v[vtx_id] = vtx3d.num_planes;

      auto& num_clusters_v    = _num_clusters_vv[vtx_id];
      auto& num_pixels_v     = _num_pixels_vv[vtx_id];
      auto& num_pixel_frac_v = _num_pixel_frac_vv[vtx_id];

      num_clusters_v.resize(3);
      num_pixels_v.resize(3);
      num_pixel_frac_v.resize(3);

      auto& vtx2d_x_v = _vtx2d_x_vv[vtx_id];
      auto& vtx2d_y_v = _vtx2d_y_vv[vtx_id];
      
      auto& circle_x_v = _circle_x_vv[vtx_id];
      auto& circle_y_v = _circle_y_vv[vtx_id];

      vtx2d_x_v.resize(3);
      vtx2d_y_v.resize(3);
    
      circle_x_v.resize(3);
      circle_y_v.resize(3);


      auto& circle_vtx_r_v     = _circle_vtx_r_vv[vtx_id];
      auto& circle_vtx_angle_v = _circle_vtx_angle_vv[vtx_id];

      circle_vtx_r_v.resize(3);
      circle_vtx_angle_v.resize(3);
      
      for(uint plane_id=0;plane_id<3;++plane_id) {

	
	const auto& circle_vtx   = vtx_cluster.get_circle_vertex(plane_id);
	const auto& circle_vtx_c = circle_vtx.center;
	
	auto& circle_x = circle_x_v[plane_id];
	auto& circle_y = circle_y_v[plane_id];

	circle_x = circle_vtx_c.x;
	circle_y = circle_vtx_c.y;

	auto& num_clusters   = num_clusters_v[plane_id];
	auto& num_pixels     = num_pixels_v[plane_id];
	auto& num_pixel_frac = num_pixel_frac_v[plane_id];

      	num_clusters   = vtx_cluster.num_clusters(plane_id);
	num_pixels     = vtx_cluster.num_pixels(plane_id);
	num_pixel_frac = vtx_cluster.num_pixel_fraction(plane_id);
	
	auto& vtx2d_x = vtx2d_x_v[plane_id];
	auto& vtx2d_y = vtx2d_y_v[plane_id];
	
	vtx2d_x = vtx3d.vtx2d_v[plane_id].pt.x;
	vtx2d_y = vtx3d.vtx2d_v[plane_id].pt.y;
	

	const auto& csetting = csarray.get_circle_setting(plane_id);
	
	auto& circle_vtx_r     = circle_vtx_r_v[plane_id];
	auto& circle_vtx_angle = circle_vtx_angle_v[plane_id];

	circle_vtx_r     = csetting._local_r;
	circle_vtx_angle = csetting._angle;
	
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
