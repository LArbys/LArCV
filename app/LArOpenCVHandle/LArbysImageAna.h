#ifndef __LARBYSIMAGEANA_H__
#define __LARBYSIMAGEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImageMaker.h"

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

    const std::vector<cv::Mat>&
    ADCImages()
    { return _adc_mat_v; }
    
    const cv::Mat&
    ADCImage(size_t planeid)
    { return _adc_mat_v[planeid]; }
    
    const std::vector<const larocv::data::Vertex3D*>&
    Verticies()
    { return *_reco_vertex_v; }

    const larocv::data::Vertex3D*
    Vertex(size_t vertexid)
    { return (*_reco_vertex_v)[vertexid]; }

    const std::vector<const larocv::data::ParticleCluster*>&
    Particles(size_t vertexid,size_t planeid)
    { return (*_particle_cluster_vvv)[vertexid][planeid]; }
    
  private:
    std::string _adc_producer;
    std::string _mc_tree_name;
    std::string _reco_tree_name;
    std::string _input_larbys_root_file;

    LArbysImageMaker _LArbysImageMaker;
    
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

    // The images
    std::vector<cv::Mat> _adc_mat_v;
    
    // Reconstructed quantities
    std::vector<const larocv::data::Vertex3D*> * _reco_vertex_v;
    std::vector<std::vector<std::vector<const larocv::data::ParticleCluster*> > >* _particle_cluster_vvv;
    std::vector<std::vector<std::vector<const larocv::data::TrackClusterCompound*> > >* _track_cluster_comp_vvv;

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

