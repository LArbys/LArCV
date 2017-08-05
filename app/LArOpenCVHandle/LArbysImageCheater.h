#ifndef __LARBYSIMAGECHEATER_H__
#define __LARBYSIMAGECHEATER_H__

#include "LArbysImage.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

namespace larcv {

  class LArbysImageCheater : public LArbysImage {

  public:
    LArbysImageCheater(const std::string name="LArbysImageCheater");
    ~LArbysImageCheater() {}

    void SetIOManager(IOManager* mgr);
    void SetTrueROIProducer(const std::string& true_prod);

  protected:
    
    bool Reconstruct(const std::vector<larcv::Image2D>& adc_image_v,
		     const std::vector<larcv::Image2D>& track_image_v,
		     const std::vector<larcv::Image2D>& shower_image_v,
		     const std::vector<larcv::Image2D>& thrumu_image_v,
		     const std::vector<larcv::Image2D>& stopmu_image_v,
    		     const std::vector<larcv::Image2D>& chstat_image_v);
    
  private:
    
    ::larutil::SpaceChargeMicroBooNE _sce;
    IOManager* _mgr;
    std::string _true_prod;
  };

  /**
     \class larcv::LArbysImageCheaterFactory
     \brief A concrete factory class for larcv::LArbysImageCheater
  */
  class LArbysImageCheaterProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageCheaterProcessFactory() { ProcessFactory::get().add_factory("LArbysImageCheater",this); }
    /// dtor
    ~LArbysImageCheaterProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImageCheater(instance_name); }
  };
  

  
}

#endif
