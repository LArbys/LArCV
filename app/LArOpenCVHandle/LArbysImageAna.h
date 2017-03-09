#ifndef __LARBYSIMAGEANA_H__
#define __LARBYSIMAGEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImageMaker.h"

namespace larcv {
  class LArbysImageAna : public ProcessBase {
  public:

    LArbysImageAna(const std::string name="LArbysImageAna");
    ~LArbysImageAna(){}
    
    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

  private:
    std::string _adc_producer;
    LArbysImageMaker _LArbysImageMaker;
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

