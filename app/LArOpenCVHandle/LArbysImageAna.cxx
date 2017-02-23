#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__
#include "LArbysImageAna.h"

#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name),
      _event_tree(nullptr),
      _vtx3d_tree(nullptr)
  {}
      
  void LArbysImageAna::configure(const PSet& cfg)
  {
    _track_vertex_estimate_algo_name = cfg.get<std::string>("TrackVertexEstimateAlgoName","");
  }
  
  void LArbysImageAna::ClearVertex() {
    _vtx2d_x_v.clear();
    _vtx2d_y_v.clear();

    _circle_x_v.clear();
    _circle_y_v.clear();

    _vtx2d_x_v.resize(3);
    _vtx2d_y_v.resize(3);

    _circle_x_v.resize(3);
    _circle_y_v.resize(3);

    _circle_xs_v.resize(3);
  }

  void LArbysImageAna::initialize()
  {
    
    _event_tree = new TTree("EventTree","");
    _event_tree->Branch("run"    ,&_run    , "run/i");
    _event_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _event_tree->Branch("event"  ,&_event  , "event/i");
    _event_tree->Branch("entry"  ,&_entry  , "entry/i");
    
    _vtx3d_tree = new TTree("Vertex3DTree","");
    _vtx3d_tree->Branch("run"    ,&_run    , "run/i");
    _vtx3d_tree->Branch("subrun" ,&_subrun , "subrun/i");
    _vtx3d_tree->Branch("event"  ,&_event  , "event/i");
    _vtx3d_tree->Branch("entry"  ,&_entry  , "entry/i");
    _vtx3d_tree->Branch("id", &_vtx3d_id, "id/i");
    _vtx3d_tree->Branch("type", &_vtx3d_type, "type/i");
    _vtx3d_tree->Branch("x", &_vtx3d_x, "x/D"  );
    _vtx3d_tree->Branch("y", &_vtx3d_y, "y/D"  );
    _vtx3d_tree->Branch("z", &_vtx3d_z, "z/D"  );
    _vtx3d_tree->Branch("vtx2d_x_v", &_vtx2d_x_v );
    _vtx3d_tree->Branch("vtx2d_y_v", &_vtx2d_y_v );
    _vtx3d_tree->Branch("cvtx2d_x_v",&_circle_x_v);
    _vtx3d_tree->Branch("cvtx2d_y_v",&_circle_y_v);
    _vtx3d_tree->Branch("cvtx2d_xs_v",&_circle_xs_v);
  }

  bool LArbysImageAna::process(IOManager& mgr)
  {
        
    LARCV_DEBUG() << "process" << std::endl;

    /// get the data manager
    const auto& data_mgr  = _mgr_ptr->DataManager();

    /// get the ass man to associate algo manager
    const auto& data_ass_mgr  = data_mgr.AssManager();
    
    /// unique event keys
    const auto& event_id = mgr.event_id();
    _run    = (uint) event_id.run();
    _subrun = (uint) event_id.subrun();
    _event  = (uint) event_id.event();
    _entry =  (uint) mgr.current_entry();

    /// get the track estimate data
    const auto vtx3d_array = (larocv::data::Vertex3DArray*)
      data_mgr.Data(data_mgr.ID(_track_vertex_estimate_algo_name), 0);
    
    auto& vtx_cluster_v = vtx3d_array->as_vector();

    _n_vtx3d = (uint) vtx_cluster_v.size();

    for(uint vtx_id=0;vtx_id<_n_vtx3d;++vtx_id) { 
      
      // clear vertex
      ClearVertex();
	
      // set the vertex index number
      _vtx3d_id=vtx_id;
      
      // get this 3D vertex
      const auto& vtx3d = vtx_cluster_v[vtx_id];
	
      // set the vertex type
      _vtx3d_type = (uint) vtx3d.type;

      // set the 3D coordinates
      _vtx3d_x = vtx3d.x;
      _vtx3d_y = vtx3d.y;
      _vtx3d_z = vtx3d.z;
      
      // set the number of planes this vertex was reconstructed from
      _vtx3d_n_planes = (uint)vtx3d.num_planes;

      for(uint plane_id=0; plane_id<3;  ++plane_id) {

	// query the vertex type it's 0 (time vtx) or 1 (wire vtx)
	if (_vtx3d_type < 2) {
	  // store circle vertex information
	  const auto& circle_vtx   = vtx3d.cvtx2d_v.at(plane_id);
	  const auto& circle_vtx_c = circle_vtx.center;
	  auto& circle_x  = _circle_x_v [plane_id];
	  auto& circle_y  = _circle_y_v [plane_id];
	  auto& circle_xs = _circle_xs_v[plane_id];
	  circle_x = circle_vtx_c.x;
	  circle_y = circle_vtx_c.y;
	  circle_xs = (uint) circle_vtx.xs_v.size();
	}

	// store the 2D vertex information for this plane
	auto& vtx2d_x = _vtx2d_x_v[plane_id];
	auto& vtx2d_y = _vtx2d_y_v[plane_id];
	
	vtx2d_x = vtx3d.vtx2d_v[plane_id].pt.x;
	vtx2d_y = vtx3d.vtx2d_v[plane_id].pt.y;
	  
      } // end plane
      _vtx3d_tree->Fill();
    } // end loop over vtx
    _event_tree->Fill();
    
    return true;
  }

  void LArbysImageAna::finalize()
  {
    _event_tree->Write();
    _vtx3d_tree->Write();
  }

}
#endif

