#ifndef __LARBYSIMAGEREANA_H__
#define __LARBYSIMAGEREANA_H__

#include "LArbysImage.h"

namespace larcv {

  class LArbysImageReAna : public LArbysImage {

  public:
    LArbysImageReAna(const std::string name="LArbysImageReAna");
    ~LArbysImageReAna() {}

    void SetIOManager(IOManager* mgr);
    void SetPGraphProducer(const std::string& pgraph_prod);
    void SetPixel2DProducer(const std::string& pixel2d_prod);
    
  protected:
    void Process();

  private:
    IOManager* _mgr;
    std::string _pgraph_prod;
    std::string _pixel2d_prod;

  };

  /**
     \class larcv::LArbysImageReAnaFactory
     \brief A concrete factory class for larcv::LArbysImageReAna
  */
  class LArbysImageReAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageReAnaProcessFactory() { ProcessFactory::get().add_factory("LArbysImageReAna",this); }
    /// dtor
    ~LArbysImageReAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImageReAna(instance_name); }
  };
  
}

#endif
