#ifndef LARBYSVERTEXFILTER_CXX
#define LARBYSVERTEXFILTER_CXX

#include "LArbysRecoHolder.h"

namespace larcv {


  std::vector<std::vector<std::pair<size_t,size_t> > >
  LArbysRecoHolder::Match(size_t vtx_id,
			  const std::vector<cv::Mat>& adc_cvimg_v) {
    
    return _vtx_ana.MatchClusters(this->PlaneParticles(vtx_id),
				  adc_cvimg_v,
				  _match_coverage,
				  _match_particles_per_plane,
				  _match_min_number);
  }

  void
  LArbysRecoHolder::Filter() {

    std::vector<const larocv::data::Vertex3D*> vertex_ptr_v;
    std::vector<std::vector<std::vector<const larocv::data::ParticleCluster*> > > particle_cluster_ptr_vvv;
    std::vector<std::vector<std::vector<const larocv::data::TrackClusterCompound*> > > track_comp_ptr_vvv;

    for(size_t vertexid=0; vertexid<this->Verticies().size(); ++vertexid) {

      if (_require_two_multiplicity) { 
	auto multiplicity=_vtx_ana.RequireParticleCount(this->PlaneParticles(vertexid),2,2);
	if (!multiplicity) continue;
      }
      if (_require_fiducial) {
	auto fiduciality=_vtx_ana.CheckFiducial(*this->Vertex(vertexid));
	if (!fiduciality) continue;
      }
      
      vertex_ptr_v.emplace_back(this->Vertex(vertexid));
      particle_cluster_ptr_vvv.emplace_back(this->PlaneParticles(vertexid));
      track_comp_ptr_vvv.emplace_back(this->PlaneTracks(vertexid));
    }

    LARCV_DEBUG()<<"Filtered "<<this->Verticies().size()<<" to "<<vertex_ptr_v.size()<<std::endl;
    
    std::swap(vertex_ptr_v,            _vertex_ptr_v);
    std::swap(particle_cluster_ptr_vvv,_particle_cluster_ptr_vvv);
    std::swap(track_comp_ptr_vvv,      _track_comp_ptr_vvv);
  }

  void
  LArbysRecoHolder::Reset() {
    //_vtx_ana.Reset();
    
    _vertex_ptr_v.clear();
    _particle_cluster_ptr_vvv.clear();
    _track_comp_ptr_vvv.clear();
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
	  if (comp_ass_id==kINVALID_SIZE && par.type==larocv::data::ParticleType_t::kTrack)
	    throw larbys("Track particle with no track!");
	  const auto& comp = comp_array->as_vector()[comp_ass_id];
	  tcluster_v[ass_id] = &comp;
	} 
	_vtx_ana.ResetPlaneInfo(mgr.InputImageMetas(0)[plane]);
      }

      _particle_cluster_ptr_vvv.emplace_back(std::move(pcluster_vv));
      _track_comp_ptr_vvv.emplace_back(std::move(tcluster_vv));
    } //end this vertex
  }    


}

#endif
