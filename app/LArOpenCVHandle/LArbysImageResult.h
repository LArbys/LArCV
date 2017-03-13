#ifndef __LARBYSIMAGERESULT_H__
#define __LARBYSIMAGERESULT_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include "LArbysImage.h"

#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"
#include "LArOpenCV/ImageCluster/AlgoData/ParticleCluster.h"
#include "LArOpenCV/ImageCluster/AlgoData/TrackClusterCompound.h"

namespace larcv {

  class LArbysImageResult : public ProcessBase {

  public:
    
    LArbysImageResult(const std::string name="LArbysImageResult");
    ~LArbysImageResult(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

    void SetManager(const::larocv::ImageClusterManager* icm)
    { _mgr_ptr = icm; }

  private:
    const ::larocv::ImageClusterManager* _mgr_ptr;
    
    std::string _combined_vertex_name;
    uint _combined_particle_offset;
  };    
  /**
     \class larcv::LArbysImageResultFactory
     \brief A concrete factory class for larcv::LArbysImageResult
  */
  class LArbysImageResultProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageResultProcessFactory() { ProcessFactory::get().add_factory("LArbysImageResult",this); }
    /// dtor
    ~LArbysImageResultProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImageResult(instance_name); }
  };

}

#endif

