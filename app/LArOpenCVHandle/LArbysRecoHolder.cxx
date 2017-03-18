#ifndef LARBYSVERTEXFILTER_CXX
#define LARBYSVERTEXFILTER_CXX

#include "LArbysRecoHolder.h"

namespace larcv {

  std::vector<std::vector<std::pair<size_t,size_t> > >
  LArbysRecoHolder::Match(size_t vtx_id,
			  const std::vector<cv::Mat>& adc_cvimg_v) {
    
    auto match = _vtx_ana.MatchClusters(this->PlaneParticles(vtx_id),
					adc_cvimg_v,
					_match_coverage,
					_match_particles_per_plane,
					_match_min_number);
    
    if (vtx_id >= _match_pvvv.size())
      _match_pvvv.resize(vtx_id+1);

    _match_pvvv[vtx_id] = match;
    return match;
  }

  void
  LArbysRecoHolder::Filter() {

    std::vector<const larocv::data::Vertex3D*> vertex_ptr_v;
    std::vector<std::vector<std::vector<const larocv::data::ParticleCluster*> > > particle_cluster_ptr_vvv;
    std::vector<std::vector<std::vector<const larocv::data::TrackClusterCompound*> > > track_comp_ptr_vvv;

    for(size_t vertexid=0; vertexid<this->Verticies().size(); ++vertexid) {

      if (_require_two_multiplicity) { 
	auto multiplicity=_vtx_ana.RequireParticleCount(this->PlaneParticles(vertexid),2,2);
	if (!multiplicity) {
	  LARCV_DEBUG() << "Filtered id " << vertexid << " @ multiplicity" << std::endl;
	  continue;
	}
      }
      if (_require_fiducial) {
	auto fiduciality=_vtx_ana.CheckFiducial(*this->Vertex(vertexid));
	if (!fiduciality) {
	  LARCV_DEBUG() << "Filtered id " << vertexid << " @ fiduciality" << std::endl;
	  continue;
	}
      }
      
      vertex_ptr_v.emplace_back(this->Vertex(vertexid));
      particle_cluster_ptr_vvv.emplace_back(this->PlaneParticles(vertexid));
      track_comp_ptr_vvv.emplace_back(this->PlaneTracks(vertexid));
    }
    
    LARCV_DEBUG() <<"Filtered "<<this->Verticies().size()<<" to "<<vertex_ptr_v.size()<<std::endl;
    std::swap(vertex_ptr_v,            _vertex_ptr_v);
    std::swap(particle_cluster_ptr_vvv,_particle_cluster_ptr_vvv);
    std::swap(track_comp_ptr_vvv,      _track_comp_ptr_vvv);
  }

  void
  LArbysRecoHolder::Reset() {
    _vertex_ptr_v.clear();
    _particle_cluster_ptr_vvv.clear();
    _track_comp_ptr_vvv.clear();
  }

  void
  LArbysRecoHolder::ResetOutput() {
    _vertex_v.clear();
    _particle_cluster_vvv.clear();
    _track_comp_vvv.clear();

    _match_pvvv.clear();
    _run=kINVALID_INT;
    _subrun=kINVALID_INT;
    _event=kINVALID_INT;
    _entry=kINVALID_INT;

    this->Reset();
  }
  void
  LArbysRecoHolder::Write() {
    if(_vertex_v.empty()) return;
    _out_tree->Fill();
  }
  
  void
  LArbysRecoHolder::Configure(const PSet& pset) {
    LARCV_DEBUG() << "start" << std::endl;

    this->set_verbosity((msg::Level_t)pset.get<int>("Verbosity",2));

    _require_two_multiplicity  = pset.get<bool>("RequireMultiplicityTwo",true);
    _require_fiducial          = pset.get<bool>("RequireFiducial",true);
    _match_coverage            = pset.get<float>("MatchCoverage",0.5);
    _match_particles_per_plane = pset.get<float>("MatchParticlesPerPlane",2);
    _match_min_number          = pset.get<float>("MatchMinimumNumber",2);

    _output_module_name   = pset.get<std::string>("OutputModuleName");
    if (_output_module_name.empty()) {
      LARCV_CRITICAL() << "Must specify output module name" << std::endl;
      throw larbys();
    }
    _output_module_offset = pset.get<size_t>("OutputModuleOffset",kINVALID_SIZE);

    LARCV_DEBUG() << "RequireMultiplicityTwo: " << _require_two_multiplicity << std::endl;
    LARCV_DEBUG() << "RequireFiducial: " << _require_fiducial << std::endl;
    LARCV_DEBUG() << "MatchCoverage: " << _match_coverage << std::endl;
    LARCV_DEBUG() << "MatchParticlesPerPlane: " << _match_particles_per_plane << std::endl;
    LARCV_DEBUG() << "MatchMinimumNumber: " << _match_min_number << std::endl;
    
    LARCV_DEBUG() << "end" << std::endl;

    _out_tree = new TTree("RecoTree","");
    _out_tree->Branch("run"   ,&_run   ,"run/i");
    _out_tree->Branch("subrun",&_subrun,"subrun/i");
    _out_tree->Branch("event" ,&_event ,"event/i");
    _out_tree->Branch("entry" ,&_entry ,"entry/i");
    _out_tree->Branch("Vertex3D_v"              ,&_vertex_v);
    _out_tree->Branch("ParticleCluster_vvv"     ,&_particle_cluster_vvv);
    _out_tree->Branch("TrackClusterCompound_vvv",&_track_comp_vvv,128000);
    _out_tree->Branch("Match_pvvv"              ,&_match_pvvv);
    return;
  }

  void
  LArbysRecoHolder::ShapeData(const larocv::ImageClusterManager& mgr) {
    const larocv::data::AlgoDataManager& data_mgr   = mgr.DataManager();
    const larocv::data::AlgoDataAssManager& ass_man = data_mgr.AssManager();

    auto output_module_id = data_mgr.ID(_output_module_name);
    if (output_module_id==kINVALID_SIZE)  {
      LARCV_CRITICAL() << "Invalid algmodule name (" << _output_module_name <<") specified" << std::endl;
      throw larbys();
    }
    const auto vtx3d_array = (larocv::data::Vertex3DArray*) data_mgr.Data(output_module_id, 0);
    const auto& vertex3d_v = vtx3d_array->as_vector();

    LARCV_DEBUG() << "Observed " << vertex3d_v.size() << " verticies" << std::endl;
    for(size_t vtxid=0;vtxid<vertex3d_v.size();++vtxid) {
      const auto& vtx3d = vertex3d_v[vtxid];
      LARCV_DEBUG() << "On vertex " << vtxid << " of type " << (uint) vtx3d.type << std::endl;
      _vertex_ptr_v.push_back(&vtx3d);

      std::vector<std::vector<const larocv::data::ParticleCluster* > > pcluster_vv;
      std::vector<std::vector<const larocv::data::TrackClusterCompound* > > tcluster_vv;
      pcluster_vv.resize(3);
      tcluster_vv.resize(3);
      
      for(size_t plane=0;plane<3;++plane) {
	
	auto& pcluster_v=pcluster_vv[plane];
	auto& tcluster_v=tcluster_vv[plane];
	
	auto output_module_id = data_mgr.ID(_output_module_name);
	
	const auto par_array = (larocv::data::ParticleClusterArray*)
	  data_mgr.Data(output_module_id, plane+_output_module_offset);

	const auto comp_array = (larocv::data::TrackClusterCompoundArray*)
	  data_mgr.Data(output_module_id, plane+_output_module_offset+3);
	
	auto par_ass_idx_v = ass_man.GetManyAss(vtx3d,par_array->ID());
	pcluster_v.resize(par_ass_idx_v.size());
	tcluster_v.resize(par_ass_idx_v.size());
	
	for(size_t ass_id=0;ass_id<par_ass_idx_v.size();++ass_id) {
	  auto ass_idx = par_ass_idx_v[ass_id];
	  if (ass_idx==kINVALID_SIZE) throw larbys("Invalid vertex->particle association detected");
	  const auto& par = par_array->as_vector()[ass_idx];
	  pcluster_v[ass_id] = &par;
	  auto comp_ass_id = ass_man.GetOneAss(par,comp_array->ID());
	  if (comp_ass_id==kINVALID_SIZE && par.type==larocv::data::ParticleType_t::kTrack) {
	    throw larbys("Track particle with no track!");
	  }
	  else if (comp_ass_id==kINVALID_SIZE) {
	    tcluster_v[ass_id] = nullptr;
	    continue;
	  }
	  const auto& comp = comp_array->as_vector()[comp_ass_id];
	  tcluster_v[ass_id] = &comp;
	} 
	_vtx_ana.ResetPlaneInfo(mgr.InputImageMetas(0)[plane]);
      }

      _particle_cluster_ptr_vvv.emplace_back(std::move(pcluster_vv));
      _track_comp_ptr_vvv.emplace_back(std::move(tcluster_vv));
    } //end this vertex
  }

  bool
  LArbysRecoHolder::WriteOut(TFile* fout) {
    if (fout && _out_tree) {
      fout->cd();
      _out_tree->Write();
      return true;
    }
    return false;
  }
  
  void
  LArbysRecoHolder::StoreEvent(size_t run, size_t subrun, size_t event, size_t entry) {

    if (!_vertex_ptr_v.size()) return;

    size_t n_old=_vertex_v.size();
    size_t n_new=_vertex_ptr_v.size();
    size_t n_total=n_old+n_new;
    _vertex_v.resize(n_total);
    _particle_cluster_vvv.resize(n_total);
    _track_comp_vvv.resize(n_total);
    
    _run    = run;
    _subrun = subrun;
    _event  = event;
    _entry  = entry;
    
    for(size_t vertexid=0;vertexid<this->Verticies().size();++vertexid) {
      
      auto& vertex              = _vertex_v.at(n_old+vertexid);
      auto& particle_cluster_vv = _particle_cluster_vvv.at(n_old+vertexid);
      auto& track_comp_vv       = _track_comp_vvv.at(n_old+vertexid);
    
      vertex = *(this->Vertex(vertexid));

      size_t nplanes=3;
      particle_cluster_vv.resize(nplanes);
      track_comp_vv.resize(nplanes);
      for(size_t plane=0;plane<nplanes;++plane) {
	auto& particle_cluster_v = particle_cluster_vv[plane];
	auto& track_comp_v = track_comp_vv[plane];
	auto npars = this->Particles(vertexid,plane).size();
	particle_cluster_v.resize(npars);
	track_comp_v.resize(npars);
	for(size_t particleid=0;particleid<npars;++particleid) {
	  auto& particle_cluster = particle_cluster_v[particleid];
	  auto& track_comp = track_comp_v[particleid];
	  particle_cluster = *(this->Particle(vertexid,plane,particleid));
	  if (!(this->Track(vertexid,plane,particleid))) {
	    LARCV_DEBUG() << "SKIP!" << std::endl;
	    continue;
	  }
	  track_comp = *(this->Track(vertexid,plane,particleid));
	} //particle id
      } //plane id
    } //vertex id
  }
}

#endif
