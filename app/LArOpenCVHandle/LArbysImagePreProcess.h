#ifndef __LARBYSIMAGEPREPROCESS_H__
#define __LARBYSIMAGEPREPROCESS_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

namespace larcv {

  class LArbysImagePreProcess : public ProcessBase {

  public:
    
    /// Default constructor
    LArbysImagePreProcess(const std::string name="LArbysImagePreProcess");
    
    /// Default destructor
    ~LArbysImagePreProcess(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    
  protected:

  };

  /**
     \class larcv::LArbysImagePreProcessFactory
     \brief A concrete factory class for larcv::LArbysImagePreProcess
  */
  class LArbysImagePreProcessProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImagePreProcessProcessFactory() { ProcessFactory::get().add_factory("LArbysImagePreProcess",this); }
    /// dtor
    ~LArbysImagePreProcessProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImagePreProcess(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

