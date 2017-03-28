#ifndef __VERTEXINROI_H__
#define __VERTEXINROI_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

namespace larcv {

  class VertexInROI : public ProcessBase {

  public:
    
    VertexInROI(const std::string name="VertexInROI");
    ~VertexInROI(){}
    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();
    
  private:
    ::larutil::SpaceChargeMicroBooNE _sce;

    std::string _truth_roi_producer;
    std::string _input_roi_producer;
    std::string _output_roi_producer;
    uint _planes_inside_threshold;
    
  };

  class VertexInROIProcessFactory : public ProcessFactoryBase {
  public:
    VertexInROIProcessFactory() { ProcessFactory::get().add_factory("VertexInROI",this); }

    ~VertexInROIProcessFactory() {}
    
    ProcessBase* create(const std::string instance_name) { return new VertexInROI(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

