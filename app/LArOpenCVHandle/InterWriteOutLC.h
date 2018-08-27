#ifndef __INTERWRITEOUTLC_H__
#define __INTERWRITEOUTLC_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

namespace larcv {

  class InterWriteOutLC : public ProcessBase {

  public:
    
    InterWriteOutLC(const std::string name="InterWriteOutLC");
    ~InterWriteOutLC(){}
    void configure(const PSet&) {}
    void initialize() {}
    bool process(IOManager& mgr);
    void finalize() {}

  };

  class InterWriteOutLCProcessFactory : public ProcessFactoryBase {
  public:
    InterWriteOutLCProcessFactory() { ProcessFactory::get().add_factory("InterWriteOutLC",this); }
    ~InterWriteOutLCProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new InterWriteOutLC(instance_name); }
  };
}
#endif

