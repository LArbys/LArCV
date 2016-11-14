#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"
#include "AlgoData/HIPClusterData.h"
#include "AlgoData/Refine2DVertexData.h"
#include "AlgoData/VertexClusterData.h"
#include "AlgoData/LinearVtxFilterData.h"
#include "AlgoData/dQdXProfilerData.h"

#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name),
      _event_tree(nullptr),
      _vtx3d_tree(nullptr),
      _particle_tree(nullptr)
  {}
    
  void LArbysImageAna::configure(const PSet& cfg)
  {
    _hipcluster_name       = cfg.get<std::string>("HIPClusterAlgoName");
    _defectcluster_name    = cfg.get<std::string>("DefectClusterAlgoName");
    _pcacandidates_name    = cfg.get<std::string>("PCACandidatesAlgoName");
    _refine2dvertex_name   = cfg.get<std::string>("Refine2DVertexAlgoName");
    _vertexcluster_name    = cfg.get<std::string>("VertexTrackClusterAlgoName");
    _linearvtxfilter_name  = cfg.get<std::string>("LinearVtxFilterAlgoName");
    _dqdxprofiler_name     = cfg.get<std::string>("dQdXProfilerAlgoName");
  }

  void LArbysImageAna::ClearParticle() {
    _num_atoms_v.clear();
    _start_x_v.clear();
    _start_y_v.clear();
    _end_x_v.clear();
    _end_y_v.clear();
    _start_end_length_v.clear();
    _atom_sum_length_v.clear();
    
  }
  
  void LArbysImageAna::ClearEvent() {
    _n_mip_ctors_v.clear();
    _n_hip_ctors_v.clear();
    _n_mip_ctors_v.resize(3);
    _n_hip_ctors_v.resize(3);
  }
  
  void LArbysImageAna::ClearVertex() {

    _vtx2d_x_v.clear();
    _vtx2d_y_v.clear();
    _circle_x_v.clear();
    _circle_y_v.clear();
    _num_clusters_v.clear();
    _num_pixels_v.clear();
    _num_pixel_frac_v.clear();
    
    _circle_vtx_r_v.clear();
    _circle_vtx_angle_v.clear();

    _vtx2d_x_v.resize(3);
    _vtx2d_y_v.resize(3);
    _circle_x_v.resize(3);
    _circle_y_v.resize(3);
    _num_clusters_v.resize(3);
    _num_pixels_v.resize(3);
    _num_pixel_frac_v.resize(3);
    _circle_vtx_r_v.resize(3);
    _circle_vtx_angle_v.resize(3);
    
  }

  void LArbysImageAna::initialize()
  {
    _event_tree = new TTree("EventTree","");

    _event_tree->Branch("run"    ,&_run    , "run/i");
    _event_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _event_tree->Branch("event"  ,&_event  , "event/i");

    /// HIP cluster data
    _event_tree->Branch("n_mip_ctors_v", &_n_mip_ctors_v);
    _event_tree->Branch("n_hip_ctors_v", &_n_hip_ctors_v);

    /// VertexTrackCluster
    _event_tree->Branch("n_vtx3d", &_n_vtx3d, "n_vtx3d/i");
    
    _vtx3d_tree = new TTree("Vtx3DTree","");

    /// Unique event keys
    _vtx3d_tree->Branch("run"    ,&_run    , "run/i");
    _vtx3d_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _vtx3d_tree->Branch("event"  ,&_event  , "event/i");

    _vtx3d_tree->Branch("vtx3d_id", &_vtx3d_id, "vtx3d_id/i");
    
    _vtx3d_tree->Branch("vtx3d_x", &_vtx3d_x, "vtx3d_x/D"  );
    _vtx3d_tree->Branch("vtx3d_y", &_vtx3d_y, "vtx3d_y/D"  );
    _vtx3d_tree->Branch("vtx3d_z", &_vtx3d_z, "vtx3d_z/D"  );

    _vtx3d_tree->Branch("vtx2d_x_v", &_vtx2d_x_v );
    _vtx3d_tree->Branch("vtx2d_y_v", &_vtx2d_y_v );

    _vtx3d_tree->Branch("circle_vtx_x_v",&_circle_x_v);
    _vtx3d_tree->Branch("circle_vtx_y_v",&_circle_y_v);

    _vtx3d_tree->Branch("num_planes"       , &_num_planes, "num_planes/i");
    
    _vtx3d_tree->Branch("num_clusters_v"   , &_num_clusters_v);
    _vtx3d_tree->Branch("num_pixels_v"     , &_num_pixels_v);
    _vtx3d_tree->Branch("num_pixel_frac_v" , &_num_pixel_frac_v);
    _vtx3d_tree->Branch("sum_pixel_frac"   ,&_sum_pixel_frac ,"sum_pixel_frac/d");
    //LinearVtxFilter
    _vtx3d_tree->Branch("circle_vtx_r_v",&_circle_vtx_r_v);
    _vtx3d_tree->Branch("circle_vtx_angle_v",&_circle_vtx_angle_v);

    //Particle Tree
    _particle_tree = new TTree("ParticleTree","");

    _particle_tree->Branch("run"    ,&_run    , "run/i");
    _particle_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _particle_tree->Branch("event"  ,&_event  , "event/i");

    _particle_tree->Branch("vtx3d_id", &_vtx3d_id, "vtx3d_id/i");
    _particle_tree->Branch("plane_id", &_plane_id, "plane_id/i");
    _particle_tree->Branch("n_pars",&_n_pars,"n_pars/i");
    _particle_tree->Branch("num_atoms_v",&_num_atoms_v);
    _particle_tree->Branch("start_x_v",&_start_x_v);
    _particle_tree->Branch("start_y_v",&_start_y_v);
    _particle_tree->Branch("end_x_v",&_end_x_v);
    _particle_tree->Branch("end_y_v",&_end_y_v);
    _particle_tree->Branch("start_end_length_v",&_start_end_length_v);
    _particle_tree->Branch("atom_sum_length_v",&_atom_sum_length_v);
    
  }
  
  bool LArbysImageAna::process(IOManager& mgr)
  {
    ClearEvent();
    
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

    /// dQdX profiler
    auto dqdxprofiler_data = (larocv::data::dQdXProfilerData*)dm.Data( dm.ID(_dqdxprofiler_name) );

    _n_vtx3d = (uint) vtx_cluster_v.size();
    
    for(uint vtx_id=0;vtx_id<_n_vtx3d;++vtx_id) {
      ClearVertex();

      _vtx3d_id=vtx_id;
      const auto& vtx_cluster = vtx_cluster_v[vtx_id];
      
      const auto& vtx3d = vtx_cluster.get_vertex();

      auto& csarray = circle_setting_array_v[vtx_id];

      const auto& pardqdxarr = dqdxprofiler_data->get_vertex_cluster(vtx_id);

      _vtx3d_x = vtx3d.x;
      _vtx3d_y = vtx3d.y;
      _vtx3d_z = vtx3d.z;

      _num_planes     = vtx3d.num_planes;
      _sum_pixel_frac = 0;
      for(uint plane_id=0;plane_id<3;++plane_id) {

	_plane_id=plane_id;
	const auto& circle_vtx   = vtx_cluster.get_circle_vertex(plane_id);
	const auto& circle_vtx_c = circle_vtx.center;
	
	auto& circle_x = _circle_x_v[plane_id];
	auto& circle_y = _circle_y_v[plane_id];

	circle_x = circle_vtx_c.x;
	circle_y = circle_vtx_c.y;

	auto& num_clusters   = _num_clusters_v[plane_id];
	auto& num_pixels     = _num_pixels_v[plane_id];
	auto& num_pixel_frac = _num_pixel_frac_v[plane_id];

      	num_clusters   = vtx_cluster.num_clusters(plane_id);
	num_pixels     = vtx_cluster.num_pixels(plane_id);
	num_pixel_frac = vtx_cluster.num_pixel_fraction(plane_id);
	
	if(num_pixel_frac>1)num_pixel_frac = 1 ;//Force the up limit of pixel fraction to be 1.
	_sum_pixel_frac+= num_pixel_frac; 
	

	auto& vtx2d_x = _vtx2d_x_v[plane_id];
	auto& vtx2d_y = _vtx2d_y_v[plane_id];
	
	vtx2d_x = vtx3d.vtx2d_v[plane_id].pt.x;
	vtx2d_y = vtx3d.vtx2d_v[plane_id].pt.y;
	

	const auto& csetting = csarray.get_circle_setting(plane_id);
	
	auto& circle_vtx_r     = _circle_vtx_r_v[plane_id];
	auto& circle_vtx_angle = _circle_vtx_angle_v[plane_id];

	circle_vtx_r     = csetting._local_r;
	circle_vtx_angle = csetting._angle;

	//list of particles on this plane
	const auto& pardqdx_v = pardqdxarr.get_cluster(plane_id);
	ClearParticle();
	_n_pars = pardqdx_v.size();
	
	_num_atoms_v.resize(_n_pars);
	_start_x_v.resize(_n_pars);
	_start_y_v.resize(_n_pars);
	_end_x_v.resize(_n_pars);
	_end_y_v.resize(_n_pars);
	_start_end_length_v.resize(_n_pars);
	_atom_sum_length_v.resize(_n_pars);
	//_atomid_v.resize(_n_pars);
	
	
	for(uint pidx=0; pidx < _n_pars; ++pidx) {
	  auto& num_atoms        = _num_atoms_v[pidx];
	  auto& start_x          = _start_x_v[pidx];
	  auto& start_y          = _start_y_v[pidx];
	  auto& end_x            = _end_x_v[pidx];
	  auto& end_y            = _end_y_v[pidx];
	  auto& start_end_length = _start_end_length_v[pidx];
	  auto& atom_sum_length  = _atom_sum_length_v[pidx];
	  //auto& atomid           = _atomid_v[pidx];
		
	  const auto& pardqdx = pardqdx_v[pidx];

	  num_atoms = pardqdx.num_atoms();
	  start_x = pardqdx.start_pt().x;
	  start_y = pardqdx.start_pt().y;
	  end_x  = pardqdx.end_pt().x;
	  end_y  = pardqdx.end_pt().y;
	  	  
	  start_end_length = geo2d::length(pardqdx.end_pt() - pardqdx.start_pt());
	  atom_sum_length=0.0;
	  
	  //loop over ordered atomics and calcluate the start end length 1-by-1, sum them
	  for(auto aid : pardqdx.atom_id_array())
	    atom_sum_length += geo2d::length(pardqdx.atom_end_pt(aid) - pardqdx.atom_start_pt(aid));
	}
	
	_particle_tree->Fill();
      }
      
      _vtx3d_tree->Fill();
    }

    _event_tree->Fill();
    

    return true;
  }

  void LArbysImageAna::finalize()
  {
    _event_tree->Write();
    _vtx3d_tree->Write();
    _particle_tree->Write();
  }

}
#endif

