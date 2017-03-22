#ifndef LARBYSVERTEXFILTER_H
#define LARBYSVERTEXFILTER_H

#include "Base/larcv_base.h"
#include "Base/PSet.h"
#include "LArOpenCV/Core/ImageManager.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterManager.h"
#include "LArOpenCV/ImageCluster/AlgoClass/VertexAnalysis.h"
#include "TTree.h"

namespace larcv {

  class LArbysRecoHolder : public larcv_base {
    
  public:

    LArbysRecoHolder() :
      _vtx_ana(),
      _out_tree(nullptr)
    {  Reset(); }
    ~LArbysRecoHolder(){}
    
    void
    FilterMatches();

    void
    ShapeData(const larocv::ImageClusterManager& mgr);
    
    void
    Filter();

    void
    Configure(const PSet& pset);    
    

    std::vector<std::vector<std::pair<size_t,size_t> > >
    Match(size_t vtx_id,
	  const std::vector<cv::Mat>& adc_cvimg_v);

    void
    Reset();
    
    void
    ResetOutput();

    void
    StoreEvent(size_t run, size_t subrun, size_t event, size_t entry);
    
    bool
    WriteOut(TFile* fout);

    void
    Write();
    
    //
    //-> Getters
    //

    const larocv::VertexAnalysis&
    ana()
    { return _vtx_ana; }
    
    const larocv::data::Vertex3D*
    Vertex(size_t vertexid)
    { return _vertex_ptr_v[vertexid]; }

    const std::vector<const larocv::data::Vertex3D*>
    Verticies()
    { return _vertex_ptr_v; }

    const std::vector<std::vector<std::vector<const larocv::data::ParticleCluster*> > >&
    VertexPlaneParticles()
    { return _particle_cluster_ptr_vvv; }
    
    const std::vector<std::vector<const larocv::data::ParticleCluster*> >&
    PlaneParticles(size_t vertexid)
    { return _particle_cluster_ptr_vvv[vertexid]; }
    
    const std::vector<const larocv::data::ParticleCluster*>&
    Particles(size_t vertexid,size_t planeid)
    { return _particle_cluster_ptr_vvv[vertexid][planeid]; }

    const larocv::data::ParticleCluster*
    Particle(size_t vertexid,size_t planeid,size_t particleid)
    { return _particle_cluster_ptr_vvv[vertexid][planeid][particleid]; }

    const std::vector<std::vector<std::vector<const larocv::data::TrackClusterCompound*> > >&
    VertexPlaneTracks(size_t vertexid)
    { return _track_comp_ptr_vvv; }
    
    const std::vector<std::vector<const larocv::data::TrackClusterCompound*> >&
    PlaneTracks(size_t vertexid)
    { return _track_comp_ptr_vvv[vertexid]; }
    
    const std::vector<const larocv::data::TrackClusterCompound*>&
    Tracks(size_t vertexid,size_t planeid)
    { return _track_comp_ptr_vvv[vertexid][planeid]; }
    
    const larocv::data::TrackClusterCompound*
    Track(size_t vertexid,size_t planeid,size_t trackid)
    { return _track_comp_ptr_vvv[vertexid][planeid][trackid]; }

  private:

    std::string _output_module_name;
    size_t _output_module_offset;
    bool _require_two_multiplicity;
    bool _require_fiducial;
    float _match_coverage;
    float _match_particles_per_plane;
    float _match_min_number;

    larocv::VertexAnalysis _vtx_ana;
    
    std::vector<const larocv::data::Vertex3D*> _vertex_ptr_v;
    std::vector<std::vector<std::vector<const larocv::data::ParticleCluster*> > > _particle_cluster_ptr_vvv;
    std::vector<std::vector<std::vector<const larocv::data::TrackClusterCompound*> > > _track_comp_ptr_vvv;

    TTree* _out_tree;
    uint _run;
    uint _subrun;
    uint _event;
    uint _entry;
    std::vector<larocv::data::Vertex3D> _vertex_v;
    std::vector<std::vector<std::vector<larocv::data::ParticleCluster> > > _particle_cluster_vvv;
    std::vector<std::vector<std::vector<larocv::data::TrackClusterCompound> > > _track_comp_vvv;
    std::vector<std::vector<std::vector<std::pair<size_t,size_t> > > > _match_pvvv;
      
  };
}

#endif

