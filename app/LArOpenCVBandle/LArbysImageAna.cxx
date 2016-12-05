#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"
#include "AlgoData/HIPClusterData.h"
#include "AlgoData/Refine2DVertexData.h"
#include "AlgoData/VertexClusterData.h"
#include "AlgoData/LinearVtxFilterData.h"
#include "AlgoData/dQdXProfilerData.h"

#include "AlgoData/LinearTrackClusterData.h"
#include "AlgoData/SingleShowerData.h"

#include "LArUtil/GeometryHelper.h"
#include "LArUtil/LArProperties.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name),
      _event_tree(nullptr),
      _vtx3d_tree(nullptr),
      _particle_tree(nullptr),
      _track_tree(nullptr),
      _shower_tree(nullptr)
  {}
    
  void LArbysImageAna::configure(const PSet& cfg)
  {
    _hipcluster_name         = cfg.get<std::string>("HIPClusterAlgoName");
    _defectcluster_name      = cfg.get<std::string>("DefectClusterAlgoName");
    _pcacandidates_name      = cfg.get<std::string>("PCACandidatesAlgoName");
    _refine2dvertex_name     = cfg.get<std::string>("Refine2DVertexAlgoName");
    _vertexcluster_name      = cfg.get<std::string>("VertexTrackClusterAlgoName");
    _linearvtxfilter_name    = cfg.get<std::string>("LinearVtxFilterAlgoName");
    _dqdxprofiler_name       = cfg.get<std::string>("dQdXProfilerAlgoName");
    _lineartrackcluster_name = cfg.get<std::string>("LinearTrackClusterAlgoName");
    _vertexsingleshower_name = cfg.get<std::string>("VertexSingleShowerAlgoName");
  }

  void LArbysImageAna::ClearParticle() {

    _qsum_v.clear();
    _npix_v.clear();
    _num_atoms_v.clear();
    _start_x_v.clear();
    _start_y_v.clear();
    _end_x_v.clear();
    _end_y_v.clear();
    _start_end_length_v.clear();
    _atom_sum_length_v.clear();
    _first_atom_cos_v.clear();
    _dqdx_vv.clear();
    _dqdx_start_idx_vv.clear();
    
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
    _circle_xs_v.resize(3);
    _num_clusters_v.resize(3);
    _num_pixels_v.resize(3);
    _num_pixel_frac_v.resize(3);
    _circle_vtx_r_v.resize(3);
    _circle_vtx_angle_v.resize(3);
    
  }

  void LArbysImageAna::ClearTracks() {

    _edge2D_1_x_v.clear();
    _edge2D_1_y_v.clear();
    _edge2D_2_x_v.clear();
    _edge2D_2_y_v.clear();

    _edge2D_1_x_v.resize(3);
    _edge2D_1_y_v.resize(3);
    _edge2D_2_x_v.resize(3);
    _edge2D_2_y_v.resize(3);
    
  }

  void LArbysImageAna::ClearShowers() {
    _start2D_x_v.clear();
    _start2D_y_v.clear();
    _dir2D_x_v.clear();
    _dir2D_y_v.clear();

    _start2D_x_v.resize(3);
    _start2D_y_v.resize(3);
    _dir2D_x_v.resize(3);
    _dir2D_y_v.resize(3);
    
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
    _vtx3d_tree->Branch("vtx3d_type", &_vtx3d_type, "vtx3d_type/i");
    
    _vtx3d_tree->Branch("vtx3d_x", &_vtx3d_x, "vtx3d_x/D"  );
    _vtx3d_tree->Branch("vtx3d_y", &_vtx3d_y, "vtx3d_y/D"  );
    _vtx3d_tree->Branch("vtx3d_z", &_vtx3d_z, "vtx3d_z/D"  );

    _vtx3d_tree->Branch("vtx2d_x_v", &_vtx2d_x_v );
    _vtx3d_tree->Branch("vtx2d_y_v", &_vtx2d_y_v );

    _vtx3d_tree->Branch("circle_vtx_x_v",&_circle_x_v);
    _vtx3d_tree->Branch("circle_vtx_y_v",&_circle_y_v);
    _vtx3d_tree->Branch("circle_vtx_xs_v",&_circle_xs_v);

    _vtx3d_tree->Branch("num_planes"       , &_num_planes, "num_planes/i");
    
    _vtx3d_tree->Branch("num_clusters_v"   , &_num_clusters_v);
    _vtx3d_tree->Branch("num_pixels_v"     , &_num_pixels_v);
    _vtx3d_tree->Branch("num_pixel_frac_v" , &_num_pixel_frac_v);
    _vtx3d_tree->Branch("sum_pixel_frac"   ,&_sum_pixel_frac ,"sum_pixel_frac/d");
    _vtx3d_tree->Branch("prod_pixel_frac"   ,&_prod_pixel_frac ,"prod_pixel_frac/d");
    
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

    _particle_tree->Branch("qsum_v",&_qsum_v);
    _particle_tree->Branch("npix_v",&_npix_v);

    _particle_tree->Branch("num_atoms_v",&_num_atoms_v);
    _particle_tree->Branch("start_x_v",&_start_x_v);
    _particle_tree->Branch("start_y_v",&_start_y_v);
    _particle_tree->Branch("end_x_v",&_end_x_v);
    _particle_tree->Branch("end_y_v",&_end_y_v);
    _particle_tree->Branch("start_end_length_v",&_start_end_length_v);
    _particle_tree->Branch("atom_sum_length_v",&_atom_sum_length_v);
    _particle_tree->Branch("first_atom_cos_v",&_first_atom_cos_v);
    _particle_tree->Branch("dqdx_vv",&_dqdx_vv);
    _particle_tree->Branch("dqdx_start_idx_vv",&_dqdx_start_idx_vv);

    /// LinearTrackCluster info
    _track_tree = new TTree("TrackTree","");
    
    _event_tree->Branch("n_trackclusters",&_n_trackclusters,"_n_trackclusters/i");

    _track_tree->Branch("run"    ,&_run    , "run/i");
    _track_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _track_tree->Branch("event"  ,&_event  , "event/i");

    _track_tree->Branch("edge2D_1_x_v", &_edge2D_1_x_v);
    _track_tree->Branch("edge2D_1_y_v", &_edge2D_1_y_v);
    _track_tree->Branch("edge2D_2_x_v", &_edge2D_2_x_v);
    _track_tree->Branch("edge2D_2_y_v", &_edge2D_2_y_v);
    
    /// VertexSingleShower info

    _shower_tree = new TTree("ShowerTree","");
    
    _event_tree->Branch("n_showerclusters",&_n_showerclusters,"_n_showerclusters/i");

    _shower_tree->Branch("run"    ,&_run    , "run/i");
    _shower_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _shower_tree->Branch("event"  ,&_event  , "event/i");

    _shower_tree->Branch("id", &_shower_id, "_shower_id/i");
    _shower_tree->Branch("ass_id", &_shower_ass_id, "_ass_id/i");
    _shower_tree->Branch("ass_type", &_shower_ass_type, "_ass_type/i");

    _shower_tree->Branch("vtx3D_x",&_shower_vtx3D_x,"_shower_vtx3D_x/d");
    _shower_tree->Branch("vtx3D_y",&_shower_vtx3D_y,"_shower_vtx3D_y/d");
    _shower_tree->Branch("vtx3D_z",&_shower_vtx3D_z,"_shower_vtx3D_z/d");
    
    _shower_tree->Branch("start2D_x_v", &_start2D_x_v);
    _shower_tree->Branch("start2D_y_v", &_start2D_y_v);
    
    _shower_tree->Branch("dir2D_x_v", &_dir2D_x_v);
    _shower_tree->Branch("dir2D_y_v", &_dir2D_y_v);
    
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
    const auto refine2d_data = (larocv::data::Refine2DVertexData*)dm.Data( dm.ID(_refine2dvertex_name) );

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

      // set the vertex ID
      _vtx3d_id=vtx_id;

      // get this VertexTrackCluster
      const auto& vtx_cluster = vtx_cluster_v[vtx_id];

      // set the vertex type
      _vtx3d_type = (uint) refine2d_data->get_type(vtx_id);

      // get this 3D vertex
      const auto& vtx3d = vtx_cluster.get_vertex();
      
      //get this circle's setting
      auto& csarray = circle_setting_array_v[vtx_id];

      //get the dqdx particle array for this vertex id
      const auto& pardqdxarr = dqdxprofiler_data->get_vertex_cluster(vtx_id);
      
      _vtx3d_x = vtx3d.x;
      _vtx3d_y = vtx3d.y;
      _vtx3d_z = vtx3d.z;

      _num_planes = (uint) vtx3d.num_planes;
      
      _sum_pixel_frac  = 0.0;
      _prod_pixel_frac = 1.0;
      
      for(uint plane_id=0; plane_id<3;  ++plane_id) {
	
	_plane_id=plane_id;
	
	const auto& circle_vtx   = vtx_cluster.get_circle_vertex(plane_id);
	const auto& circle_vtx_c = circle_vtx.center;
	
	auto& circle_x  = _circle_x_v [plane_id];
	auto& circle_y  = _circle_y_v [plane_id];
	auto& circle_xs = _circle_xs_v[plane_id];
	  
	circle_x = circle_vtx_c.x;
	circle_y = circle_vtx_c.y;

	circle_xs = (uint) circle_vtx.xs_v.size();
	  
	auto& num_clusters   = _num_clusters_v[plane_id];
	auto& num_pixels     = _num_pixels_v[plane_id];
	auto& num_pixel_frac = _num_pixel_frac_v[plane_id];

      	num_clusters   = vtx_cluster.num_clusters(plane_id);
	num_pixels     = vtx_cluster.num_pixels(plane_id);
	num_pixel_frac = vtx_cluster.num_pixel_fraction(plane_id);
	
	_sum_pixel_frac  += num_pixel_frac;
	_prod_pixel_frac *= num_pixel_frac; 
	
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

	//particle list from vertextrackcluster
	const auto& parcluster_v = vtx_cluster.get_clusters(plane_id);
	
	ClearParticle();
	
	_n_pars = pardqdx_v.size();
	_num_atoms_v.resize(_n_pars);
	_start_x_v.resize(_n_pars);
	_start_y_v.resize(_n_pars);
	_end_x_v.resize(_n_pars);
	_end_y_v.resize(_n_pars);
	_start_end_length_v.resize(_n_pars);
	_atom_sum_length_v.resize(_n_pars);
	_first_atom_cos_v.resize(_n_pars);
	_qsum_v.resize(_n_pars);
	_npix_v.resize(_n_pars);
	_dqdx_vv.resize(_n_pars);
	_dqdx_start_idx_vv.resize(_n_pars);
	
	for(uint pidx=0; pidx < _n_pars; ++pidx) {
	  
	  auto& num_atoms        = _num_atoms_v[pidx];
	  auto& start_x          = _start_x_v[pidx];
	  auto& start_y          = _start_y_v[pidx];
	  auto& end_x            = _end_x_v[pidx];
	  auto& end_y            = _end_y_v[pidx];
	  auto& start_end_length = _start_end_length_v[pidx];
	  auto& atom_sum_length  = _atom_sum_length_v[pidx];
	  auto& first_atom_cos   = _first_atom_cos_v[pidx];
	  auto& qsum             = _qsum_v[pidx];
	  auto& npix             = _npix_v[pidx];

	  //this is a special from VertexTrackCluster
	  const auto& parcluster = parcluster_v[pidx];

	  npix = parcluster._num_pixel;
	  qsum = parcluster._qsum;
	    
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

	  //use the first atomic to estimate the direction, get the first atomic end point location
	  //calculate direction from 2D vertex
	  const auto& first_atom_end = pardqdx.atom_end_pt(0);
	  
	  auto dir=first_atom_end-geo2d::Vector<float>(vtx2d_x,vtx2d_y);
	  double cosangle = dir.x / sqrt(dir.x*dir.x + dir.y*dir.y);
	  
	  first_atom_cos = cosangle;
	  
	  auto& dqdx_v = _dqdx_vv[pidx];
	  dqdx_v = pardqdx.dqdx(); // copy it.

	  auto& dqdx_start_idx_v = _dqdx_start_idx_vv[pidx];
	  dqdx_start_idx_v.resize(num_atoms);

	  for(uint atom_id=0;atom_id<num_atoms;++atom_id)
	    dqdx_start_idx_v[atom_id] = (uint) pardqdx.atom_start_index(atom_id);
	  
	}
	
	_particle_tree->Fill();
      }
      
      _vtx3d_tree->Fill();
    }


    /// LinearTrackCluster
    auto lineartrackcluster_data = (larocv::data::LinearTrackArray*) dm.Data( dm.ID(_lineartrackcluster_name) );
    const auto& track_clusters = lineartrackcluster_data->get_clusters();

    _n_trackclusters = (uint) track_clusters.size();

    for( uint trk_cluster_idx=0; trk_cluster_idx < track_clusters.size(); trk_cluster_idx++) {

      ClearTracks();
      
      const auto& trk_cluster = lineartrackcluster_data->get_cluster(trk_cluster_idx);
      
      for( uint plane_id=0; plane_id<3; ++plane_id) {

	auto& edge2D_1_x = _edge2D_1_x_v[plane_id];
	auto& edge2D_1_y = _edge2D_1_y_v[plane_id];
	auto& edge2D_2_x = _edge2D_2_x_v[plane_id];
	auto& edge2D_2_y = _edge2D_2_y_v[plane_id];
	
	const auto& trk_linear2d = trk_cluster.get_cluster(plane_id);

	const auto& edge1 = trk_linear2d.edge1;
	const auto& edge2 = trk_linear2d.edge2;
	
	edge2D_1_x = edge1.x;
	edge2D_1_y = edge1.y;
	edge2D_2_x = edge2.x;
	edge2D_2_y = edge2.y;
	
      }
      
      _track_tree->Fill();
    }

    
    /// VertexSingleShower
    auto vertexsingleshower_data = (larocv::data::SingleShowerArray*) dm.Data( dm.ID(_vertexsingleshower_name) );
    const auto& shower_clusters = vertexsingleshower_data->get_showers();
    
    _n_showerclusters = (uint) shower_clusters.size();

    for( uint shr_cluster_idx=0; shr_cluster_idx < shower_clusters.size(); shr_cluster_idx++) {

      ClearShowers();
      
      const auto& shr_cluster = shower_clusters[shr_cluster_idx];

      _shower_id       = shr_cluster.id();
      _shower_ass_id   = shr_cluster.ass_id();
      _shower_ass_type = shr_cluster.ass_type();

      const auto& shower_vtx3d = shr_cluster.get_vertex();
      _shower_vtx3D_x = shower_vtx3d.x;
      _shower_vtx3D_y = shower_vtx3d.y;
      _shower_vtx3D_z = shower_vtx3d.z;

      for( uint plane_id=0; plane_id<3; ++plane_id) {

	auto& start2D_x = _start2D_x_v[plane_id];
	auto& start2D_y = _start2D_y_v[plane_id];
	auto& dir2D_x   = _dir2D_x_v[plane_id];
	auto& dir2D_y   = _dir2D_y_v[plane_id];
	
	const auto& shr_cluster2d = shr_cluster.get_cluster(plane_id);

	const auto& start = shr_cluster2d.start;
	const auto& dir   = shr_cluster2d.dir;

	start2D_x = start.x;
	start2D_y = start.y;

	dir2D_x   = dir.x;
	dir2D_y   = dir.y;
	
      }
      
      _shower_tree->Fill();
    }
    
    _event_tree->Fill();
    
    return true;
  }

  void LArbysImageAna::finalize()
  {
    _event_tree->Write();
    _vtx3d_tree->Write();
    _particle_tree->Write();
    _track_tree->Write();
    _shower_tree->Write();
  }

}
#endif

