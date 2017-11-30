#ifndef __ROICLIPPER_H__
#define __ROICLIPPER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

namespace larcv {

  class ROIClipper : public ProcessBase {

  public:
    ROIClipper(const std::string name="ROIClipper");
    ~ROIClipper(){}
    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

  private:
    std::string _img_producer;       
    std::string _input_roi_producer; 
    std::string _output_roi_producer;
    bool _remove_duplicates;

  };

  class ROIClipperProcessFactory : public ProcessFactoryBase {

  public:
    ROIClipperProcessFactory() { ProcessFactory::get().add_factory("ROIClipper",this); }
    ~ROIClipperProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new ROIClipper(instance_name); }
  };

}
#endif
