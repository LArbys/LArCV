#ifndef __LARBYSIMAGEANA_H__
#define __LARBYSIMAGEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"
#include "LArOpenCV/ImageCluster/AlgoData/ParticleCluster.h"
#include "LArOpenCV/ImageCluster/AlgoData/TrackClusterCompound.h"

namespace larcv {
  class LArbysImageAna : public ProcessBase {
  public:

    LArbysImageAna(const std::string name="LArbysImageAna");
    ~LArbysImageAna(){}
    
    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

    void SetInputLArbysFile(std::string file)
    { _input_larbys_root_file = file; }

    const larocv::data::Vertex3D&
    TrueVertex()
    { return _mc_vertex; }
    
    const std::vector<larocv::data::Vertex3D>&
    Verticies()
    { return *_reco_vertex_v; }

    const larocv::data::Vertex3D&
    Vertex(size_t vertexid)
    { return (*_reco_vertex_v)[vertexid]; }

    const std::vector<larocv::data::ParticleCluster>&
    Particles(size_t vertexid,size_t planeid)
    { return (*_particle_cluster_vvv)[vertexid][planeid]; }

    const std::vector<larocv::data::TrackClusterCompound>&
    Tracks(size_t vertexid,size_t planeid)
    { return (*_track_cluster_comp_vvv)[vertexid][planeid]; }
    
  private:

    std::string _mc_tree_name;
    std::string _reco_tree_name;
    std::string _input_larbys_root_file;

    TChain* _mc_chain;
    TChain* _reco_chain;

    size_t _mc_entries;
    size_t _reco_entries;

    uint _mc_event;
    uint _mc_entry;
    uint _mc_run;
    uint _mc_subrun;

    uint _reco_event;
    uint _reco_entry;
    uint _reco_run;
    uint _reco_subrun;

    size_t _mc_index;
    size_t _reco_index;

    bool increment(uint entry);
    
    // Reconstructed quantities
    std::vector<larocv::data::Vertex3D> * _reco_vertex_v;
    std::vector<std::vector<std::vector<larocv::data::ParticleCluster> > >* _particle_cluster_vvv;
    std::vector<std::vector<std::vector<larocv::data::TrackClusterCompound> > >* _track_cluster_comp_vvv;

    // MC quantities
    larocv::data::Vertex3D _mc_vertex;
    double _true_x;
    double _true_y;
    double _true_z;
    std::vector<double>* _vtx2d_w_v;
    std::vector<double>* _vtx2d_t_v;
    
  };

  class LArbysImageAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageAnaProcessFactory() { ProcessFactory::get().add_factory("LArbysImageAna",this); }
    /// dtor
    ~LArbysImageAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImageAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

