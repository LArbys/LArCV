#ifndef __INTERWRITEOUT_H__
#define __INTERWRITEOUT_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

namespace larcv {

  class InterWriteOut : public ProcessBase {

  public:
    
    InterWriteOut(const std::string name="InterWriteOut");
    ~InterWriteOut(){}
    void configure(const PSet&) {}
    void initialize() {}
    bool process(IOManager& mgr);
    void finalize() {}

  };

  class InterWriteOutProcessFactory : public ProcessFactoryBase {
  public:
    InterWriteOutProcessFactory() { ProcessFactory::get().add_factory("InterWriteOut",this); }
    ~InterWriteOutProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new InterWriteOut(instance_name); }
  };
}
#endif

