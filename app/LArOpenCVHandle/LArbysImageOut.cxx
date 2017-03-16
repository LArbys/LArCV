#ifndef __LARBYSIMAGEOUT_CXX__
#define __LARBYSIMAGEOUT_CXX__
#include "LArbysImageOut.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"
#include "CVUtil/CVUtil.h"

namespace larcv {

  LArbysImageOut::LArbysImageOut(const std::string name)
    : LArbysImageAnaBase(name),
      _event_tree(nullptr),
      _vtx3d_tree(nullptr),
      _vtx_ana()
  {}
      
  void LArbysImageOut::Configure(const PSet& cfg)
  {
    _combined_vertex_name = cfg.get<std::string>("CombinedVertexName");
    _combined_particle_offset = cfg.get<uint>("ParticleOffset");
  }
  
  void LArbysImageOut::ClearVertex() {
    _vtx2d_x_v.clear();
    _vtx2d_y_v.clear();
    _circle_x_v.clear();
    _circle_y_v.clear();
    _vtx2d_x_v.resize(3);
    _vtx2d_y_v.resize(3);
    _circle_x_v.resize(3);
    _circle_y_v.resize(3);
    _circle_xs_v.resize(3);
    _par_multi.clear();
    _par_multi.resize(3);
    _ntrack_par_v.clear();
    _nshower_par_v.clear();
    _ntrack_par_v.resize(3);
    _nshower_par_v.resize(3);
    std::fill(_ntrack_par_v.begin(), _ntrack_par_v.end(), 0);
    std::fill(_nshower_par_v.begin(), _nshower_par_v.end(), 0);
  }

  void LArbysImageOut::Initialize()
  {
    
    _event_tree = new TTree("EventTree","");
    _event_tree->Branch("run",&_run,"run/i");
    _event_tree->Branch("subrun",&_subrun,"subrun/i");
    _event_tree->Branch("event",&_event,"event/i");
    _event_tree->Branch("entry",&_entry,"entry/i");
    _event_tree->Branch("Vertex3D_v",&_vertex3d_v);
    _event_tree->Branch("ParticleCluster_vvv",&_particle_cluster_vvv);
    _event_tree->Branch("TrackClusterCompound_vvv",&_track_compound_vvv);
    /*
    _event_tree->Branch("particle_vv",&_particle_vv);
    _event_tree->Branch("particle_start_vv",&_particle_start_vv);
    _event_tree->Branch("particle_end_vv",&_particle_end_vv);
    _event_tree->Branch("particle_start2d_vvv",&_particle_start2d_vvv);
    _event_tree->Branch("particle_end2d_vvv",&_particle_end2d_vvv);
    */
    
    _vtx3d_tree = new TTree("Vertex3DTree","");
    _vtx3d_tree->Branch("run",&_run,"run/i");
    _vtx3d_tree->Branch("subrun",&_subrun,"subrun/i");
    _vtx3d_tree->Branch("event",&_event,"event/i");
    _vtx3d_tree->Branch("entry",&_entry,"entry/i");
    _vtx3d_tree->Branch("id",&_vtx3d_id,"id/i");
    _vtx3d_tree->Branch("type",&_vtx3d_type,"type/i");
    _vtx3d_tree->Branch("x",&_vtx3d_x,"x/D");
    _vtx3d_tree->Branch("y",&_vtx3d_y,"y/D");
    _vtx3d_tree->Branch("z",&_vtx3d_z,"z/D");
    _vtx3d_tree->Branch("vtx2d_x_v", &_vtx2d_x_v );
    _vtx3d_tree->Branch("vtx2d_y_v", &_vtx2d_y_v );
    _vtx3d_tree->Branch("cvtx2d_x_v",&_circle_x_v);
    _vtx3d_tree->Branch("cvtx2d_y_v",&_circle_y_v);
    _vtx3d_tree->Branch("cvtx2d_xs_v",&_circle_xs_v);
    _vtx3d_tree->Branch("multi_v",&_par_multi);
    _vtx3d_tree->Branch("ntrack_par_v",&_ntrack_par_v);
    _vtx3d_tree->Branch("nshower_par_v",&_nshower_par_v);
    /*
      _event_tree->Branch("Vertex3DArray",&_vertex3d_array);
      _event_tree->Branch("ParticleClusterArray_v",&_particle_cluster_array_v);
      _event_tree->Branch("TrackClusterCompoundArray_v",&_track_cluster_compound_array_v);
      _event_tree->Branch("AlgoDataAssManager",&_ass_man);
    */
  }

  bool LArbysImageOut::Analyze(const larocv::ImageClusterManager& mgr)
  {
    LARCV_DEBUG() << "process" << std::endl;
    
    /// get the data manager
    const larocv::data::AlgoDataManager& data_mgr   = mgr.DataManager();
    const larocv::data::AlgoDataAssManager& ass_man = data_mgr.AssManager();
    
    /*
      _ass_man = ass_man;
    */
    
    /// get the track estimate data
    const auto vtx3d_array = (larocv::data::Vertex3DArray*)
      data_mgr.Data(data_mgr.ID(_combined_vertex_name), 0);

    /*
      _vertex3d_array = *vtx3d_array;
      _particle_cluster_array_v.resize(3,nullptr);
      _track_cluster_compound_array_v.resize(3,nullptr);
    */
    
    const auto& vertex3d_v = vtx3d_array->as_vector();

    _n_vtx3d = vertex3d_v.size();

    _vertex3d_v.clear();
    _vertex3d_v.resize(vertex3d_v.size());
    _particle_cluster_vvv.clear();
    _particle_cluster_vvv.resize(vertex3d_v.size());
    _track_compound_vvv.clear();
    _track_compound_vvv.resize(vertex3d_v.size());


    for(uint vtxid=0;vtxid<_n_vtx3d;++vtxid) { 
      
      auto& particle_cluster_vv = _particle_cluster_vvv[vtxid];
      auto& track_compound_vv = _track_compound_vvv[vtxid];

      ClearVertex();
      
      // set the vertex index number
      _vtx3d_id=vtxid;
      
      // get this 3D vertex
      const auto& vtx3d = vertex3d_v[vtxid];

      // store this vertex
      _vertex3d_v[vtxid] = vtx3d;
      
      // set the vertex type
      _vtx3d_type = (uint) vtx3d.type;
      
      // set the 3D coordinates
      _vtx3d_x = vtx3d.x;
      _vtx3d_y = vtx3d.y;
      _vtx3d_z = vtx3d.z;
      
      // set the number of planes this vertex was reconstructed from
      _vtx3d_n_planes = (uint)vtx3d.num_planes;

      particle_cluster_vv.resize(3);
      track_compound_vv.resize(3);
      
      for(uint plane=0; plane<3;  ++plane) {
    	auto& particle_cluster_v = particle_cluster_vv[plane];
    	auto& track_compound_v = track_compound_vv[plane];

    	/*
	  auto& particle_cluster_array = _particle_cluster_array_v[plane];
	  auto& track_cluster_compound_array = _track_cluster_compound_array_v[plane];
	*/
	
    	auto track_particle_cluster_id = data_mgr.ID(_combined_vertex_name);
    	// query the vertex type it's 0 (time vtx) or 1 (wire vtx)
    	if (_vtx3d_type < 2) {
    	  // store circle vertex information
    	  const auto& circle_vtx   = vtx3d.cvtx2d_v.at(plane);
    	  const auto& circle_vtx_c = circle_vtx.center;
    	  auto& circle_x  = _circle_x_v [plane];
    	  auto& circle_y  = _circle_y_v [plane];
    	  auto& circle_xs = _circle_xs_v[plane];
    	  circle_x = circle_vtx_c.x;
    	  circle_y = circle_vtx_c.y;
    	  circle_xs = (uint) circle_vtx.xs_v.size();
    	}
	
    	auto& vtx2d_x = _vtx2d_x_v[plane];
    	auto& vtx2d_y = _vtx2d_y_v[plane];

    	// store the 2D vertex information for this plane
    	vtx2d_x = vtx3d.cvtx2d_v[plane].center.x;
    	vtx2d_y = vtx3d.cvtx2d_v[plane].center.y;

    	//get the particle cluster array
    	const auto par_array = (larocv::data::ParticleClusterArray*)
    	  data_mgr.Data(track_particle_cluster_id, plane+_combined_particle_offset);

    	//get the compound array
    	const auto comp_array = (larocv::data::TrackClusterCompoundArray*)
    	  data_mgr.Data(track_particle_cluster_id, plane+_combined_particle_offset+3);
	
    	auto par_ass_idx_v = ass_man.GetManyAss(vtx3d,par_array->ID());

    	_par_multi[plane] = (uint)par_ass_idx_v.size();

    	particle_cluster_v.resize(par_ass_idx_v.size());
    	track_compound_v.resize(par_ass_idx_v.size());
	 
    	/*
	  particle_cluster_array = par_array;
	  track_cluster_compound_array = comp_array;
	*/
	
    	for(size_t ass_id=0;ass_id<par_ass_idx_v.size();++ass_id) {
    	  auto ass_idx = par_ass_idx_v[ass_id];
    	  if (ass_idx==kINVALID_SIZE) throw larbys("Invalid vertex->particle association detected");
    	  const auto& par = par_array->as_vector()[ass_idx];
    	  particle_cluster_v[ass_id] = par; // yes, copy it bro
    	  if (par.type==larocv::data::ParticleType_t::kTrack) _ntrack_par_v[plane]++;
    	  if (par.type==larocv::data::ParticleType_t::kShower)_nshower_par_v[plane]++;
    	  auto comp_ass_id = ass_man.GetOneAss(par,comp_array->ID());
    	  if (comp_ass_id==kINVALID_SIZE) continue;
    	  const auto& comp = comp_array->as_vector()[comp_ass_id];
    	  track_compound_v[ass_id] = comp;
    	}
      } // end plane
      _vtx3d_tree->Fill();
    } //end loop over vertex
    _event_tree->Fill();

    //For adrien...
    /*
    auto& adc_img_v=mgr.InputImages(0);
    auto& adc_meta_v=mgr.InputImageMetas(0);

    for(size_t vtxid=0;vtxid<vertex3d_v.size();++vtxid) { 
      
      const auto& vtx3d = vertex3d_v[vtxid];
      
      std::vector<std::vector<const larocv::data::ParticleCluster* > > pcluster_vv;
      std::vector<std::vector<const larocv::data::TrackClusterCompound* > > tcluster_vv;
      
      pcluster_vv.resize(3);
      tcluster_vv.resize(3);
      
      for(uint plane=0; plane<3;  ++plane) {

	auto& pcluster_v=pcluster_vv[plane];
	auto& tcluster_v=tcluster_vv[plane];
	
	auto track_particle_cluster_id = data_mgr.ID(_combined_vertex_name);
	
	const auto par_array = (larocv::data::ParticleClusterArray*)
	  data_mgr.Data(track_particle_cluster_id, plane+_combined_particle_offset);

	const auto comp_array = (larocv::data::TrackClusterCompoundArray*)
	  data_mgr.Data(track_particle_cluster_id, plane+_combined_particle_offset+3);
	
	auto par_ass_idx_v = ass_man.GetManyAss(vtx3d,par_array->ID());
	pcluster_v.resize(par_ass_idx_v.size());
	tcluster_v.resize(par_ass_idx_v.size());
	
	for(size_t ass_id=0;ass_id<par_ass_idx_v.size();++ass_id) {
	  auto ass_idx = par_ass_idx_v[ass_id];
	  if (ass_idx==kINVALID_SIZE) throw larbys("Invalid vertex->particle association detected");
	  const auto& par = par_array->as_vector()[ass_idx];
	  pcluster_v[ass_id] = &par;
	  auto comp_ass_id = ass_man.GetOneAss(par,comp_array->ID());
	  if (comp_ass_id==kINVALID_SIZE) continue;//throw larbys("Bad compound ID");
	  const auto& comp = comp_array->as_vector()[comp_ass_id];
	  tcluster_v[ass_id] = &comp;
	} // end particle
	
	_vtx_ana.ResetPlaneInfo(adc_meta_v[plane]);
      } //end plane
      //}
    //_event_tree->Fill();
    */
    /*
    //do the matching
    auto match_vv = _vtx_ana.MatchClusters(pcluster_vv,adc_img_v,0.5,2,2);
    
      std::vector<EventImage2D> particle_v;
	
      std::vector<larcv::Vertex> particle_start_v;
      std::vector<larcv::Vertex> particle_end_v;
      std::vector<std::vector<larcv::Vertex> > particle_start2d_vv;
      std::vector<std::vector<larcv::Vertex> > particle_end2d_vv;
      
      for( auto match_v : match_vv ) {
	//for this match
	if (match_v.size()==2) {
	  std::cout << "2 plane match found" << std::endl;
	  auto& plane0 = match_v[0].first;
	  auto& id0    = match_v[0].second;
	  auto& plane1 = match_v[1].first;
	  auto& id1    = match_v[1].second;

	  const auto& cvimg0 = adc_img_v[plane0];
	  const auto& cvimg1 = adc_img_v[plane1];

	  const auto& cvmeta0 = adc_meta_v[plane0];
	  const auto& cvmeta1 = adc_meta_v[plane1];
	  
	  const auto& par0   = *(pcluster_vv[plane0][id0]);
	  const auto& par1   = *(pcluster_vv[plane1][id1]);
	  
	  const auto& track0 = *(tcluster_vv[plane0][id0]);
	  const auto& track1 = *(tcluster_vv[plane1][id1]);
	  
	  auto end0 = track0.end_pt();
	  auto end1 = track1.end_pt();

	  larocv::data::Vertex3D vertex;
	  _vtx_ana.Geo().YZPoint(end0,plane0,end1,plane1,vertex);

	  larcv::Vertex startpt3D;
	  startpt3D.Reset(vtx3d.x,
			  vtx3d.y,
			  vtx3d.z,
			  kINVALID_DOUBLE);
	  particle_start_v.emplace_back(startpt3D);
	  
	  larcv::Vertex endpt3D;
	  endpt3D.Reset(vertex.x,
			vertex.y,
			vertex.z,
			kINVALID_DOUBLE);
	  particle_end_v.emplace_back(endpt3D);

	  std::vector<larcv::Vertex> startpt2D_v;
	  startpt2D_v.resize(3);
	  std::vector<larcv::Vertex> endpt2D_v;
	  endpt2D_v.resize(3);
	  
	  for(size_t plane=0;plane<3;++plane) { 
	    auto& startpt = startpt2D_v[plane];
	    startpt.Reset(512-vtx3d.vtx2d_v[plane].pt.x,
			  vtx3d.vtx2d_v[plane].pt.y,
			  kINVALID_DOUBLE,
			  kINVALID_DOUBLE);
	    
	    auto& endpt = endpt2D_v[plane];
	    endpt.Reset(512-vertex.vtx2d_v[plane].pt.x,
			vertex.vtx2d_v[plane].pt.y,
			kINVALID_DOUBLE,
			kINVALID_DOUBLE);
	  }

	  particle_end2d_vv.emplace_back(std::move(endpt2D_v));
	  particle_start2d_vv.emplace_back(std::move(startpt2D_v));

	  auto cvimg0_m = larocv::MaskImage(cvimg0,par0._ctor,0,false);
	  auto cvimg1_m = larocv::MaskImage(cvimg1,par1._ctor,0,false);
	  
	  auto img2d0 = mat_to_image2d(cvimg0_m);
	  auto img2d1 = mat_to_image2d(cvimg1_m);

	  EventImage2D ev_img;
	  std::vector<Image2D> img2d_v;
	  img2d_v.resize(3);
	  img2d_v[plane0]=std::move(img2d0);
	  img2d_v[plane1]=std::move(img2d1);
	  ev_img.Emplace(std::move(img2d_v));
	  particle_v.emplace_back(std::move(ev_img));
	}
	
	if (match_v.size()==3) {
	  std::cout << "3 plane match found" << std::endl;
	  auto& plane0 = match_v[0].first;
	  auto& id0    = match_v[0].second;
	  auto& plane1 = match_v[1].first;
	  auto& id1    = match_v[1].second;
	  auto& plane2 = match_v[2].first;
	  auto& id2    = match_v[2].second;


	  const auto& cvimg0 = adc_img_v[plane0];
	  const auto& cvimg1 = adc_img_v[plane1];
	  const auto& cvimg2 = adc_img_v[plane2];

	  const auto& cvmeta0 = adc_meta_v[plane0];
	  const auto& cvmeta1 = adc_meta_v[plane1];
	  const auto& cvmeta2 = adc_meta_v[plane2];
	  
	  const auto& par0   = *(pcluster_vv[plane0][id0]);
	  const auto& par1   = *(pcluster_vv[plane1][id1]);
	  const auto& par2   = *(pcluster_vv[plane2][id2]);
	  
	  const auto& track0 = *(tcluster_vv[plane0][id0]);
	  const auto& track1 = *(tcluster_vv[plane1][id1]);
	  const auto& track2 = *(tcluster_vv[plane2][id2]);
	  
	  auto end0 = track0.end_pt();
	  auto end1 = track1.end_pt();
	  auto end2 = track2.end_pt();
	  
	  larocv::data::Vertex3D vertex;
	  bool found = _vtx_ana.Geo().YZPoint(end0,plane0,end1,plane1,vertex);
	  std::cout << "Testing end0 @ " << end0 << " on plane " << plane0 << " & end1 " << end1 << " @ plane " << plane1 << std::endl;
	  if (!found) { 
	    found = _vtx_ana.Geo().YZPoint(end0,plane0,end2,plane2,vertex);
	    std::cout << "Testing end0 @ " << end0 << " on plane " << plane0 << " & end2 " << end2 << " @ plane " << plane2 << std::endl;
	  }
	  if (!found) {
	    found = _vtx_ana.Geo().YZPoint(end1,plane1,end2,plane2,vertex);
	    std::cout << "Testing end1 @ " << end1 << " on plane " << plane1 << " & end2 " << end2 << " @ plane " << plane2 << std::endl;
	  }

	  if (!found) throw larbys("NOT FOUND!!");

	  larcv::Vertex startpt3D;
	  startpt3D.Reset(vtx3d.x,
			  vtx3d.y,
			  vtx3d.z,
			  kINVALID_DOUBLE);
	  particle_start_v.emplace_back(startpt3D);
	  
	  larcv::Vertex endpt3D;
	  endpt3D.Reset(vertex.x,
			vertex.y,
			vertex.z,
			kINVALID_DOUBLE);
	  particle_end_v.emplace_back(endpt3D);

	  std::vector<larcv::Vertex> startpt2D_v;
	  startpt2D_v.resize(3);
	  std::vector<larcv::Vertex> endpt2D_v;
	  endpt2D_v.resize(3);
	  
	  for(size_t plane=0;plane<3;++plane) { 
	    auto& startpt = startpt2D_v[plane];
	    startpt.Reset(512-vtx3d.vtx2d_v[plane].pt.x,
			  vtx3d.vtx2d_v[plane].pt.y,
			  kINVALID_DOUBLE,
			  kINVALID_DOUBLE);
	    
	    auto& endpt = endpt2D_v[plane];
	    endpt.Reset(512-vertex.vtx2d_v[plane].pt.x,
			vertex.vtx2d_v[plane].pt.y,
			kINVALID_DOUBLE,
			kINVALID_DOUBLE);
	  }

	  particle_end2d_vv.emplace_back(std::move(endpt2D_v));
	  particle_start2d_vv.emplace_back(std::move(startpt2D_v));

	  auto cvimg0_m = larocv::MaskImage(cvimg0,par0._ctor,0,false);
	  auto cvimg1_m = larocv::MaskImage(cvimg1,par1._ctor,0,false);
	  auto cvimg2_m = larocv::MaskImage(cvimg2,par2._ctor,0,false);

	  cv::flip(cvimg0_m,cvimg0_m,1);
	  cv::flip(cvimg1_m,cvimg1_m,1);
	  cv::flip(cvimg2_m,cvimg2_m,1);
	  
	  cv::transpose(cvimg0_m,cvimg0_m);
	  cv::transpose(cvimg1_m,cvimg1_m);
	  cv::transpose(cvimg2_m,cvimg2_m);
	  
	  auto img2d0 = mat_to_image2d(cvimg0_m);
	  auto img2d1 = mat_to_image2d(cvimg1_m);
	  auto img2d2 = mat_to_image2d(cvimg2_m);

	  EventImage2D ev_img;
	  std::vector<Image2D> img2d_v;
	  img2d_v.resize(3);
	  img2d_v[plane0]=std::move(img2d0);
	  img2d_v[plane1]=std::move(img2d1);
	  img2d_v[plane2]=std::move(img2d2);
	  ev_img.Emplace(std::move(img2d_v));
	  particle_v.emplace_back(std::move(ev_img));
	}
	else {
	  throw larbys();
	}
      } // end particles
      _particle_vv.emplace_back(std::move(particle_v));
      _particle_start_vv.emplace_back(std::move(particle_start_v));
      _particle_end_vv.emplace_back(std::move(particle_end_v));
      _particle_start2d_vvv.emplace_back(std::move(particle_start2d_vv));
      _particle_end2d_vvv.emplace_back(std::move(particle_end2d_vv));
    } // end vertex

    _event_tree->Fill();
    _particle_vv.clear();
    _particle_start_vv.clear();
    _particle_end_vv.clear();
    _particle_start2d_vvv.clear();
    _particle_end2d_vvv.clear();
    */
    
    return true;
  }
  
  void LArbysImageOut::Finalize(TFile* fout)
  {
    if(fout) {
      _event_tree->Write();
      _vtx3d_tree->Write();
    }
  }
  
}
#endif

