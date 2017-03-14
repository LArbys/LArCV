#ifndef __LARBYSIMAGEEXTRACT_H__
#define __LARBYSIMAGEEXTRACT_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImageMaker.h"

namespace larcv {

  class LArbysImageExtract : public ProcessBase {

  public:
    
    LArbysImageExtract(const std::string name="LArbysImageExtract");
    ~LArbysImageExtract(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();


    const std::vector<cv::Mat>&
    ADCImages()
    { return _adc_mat_v; }

    const std::vector<larocv::ImageMeta>&
    ADCMetas()
    { return _adc_meta_v; }
    
    const cv::Mat&
    ADCImage(size_t planeid)
    { return _adc_mat_v[planeid]; }

    const larocv::ImageMeta&
    ADCMeta(size_t planeid)
    { return _adc_meta_v[planeid]; }
    
    const std::vector<cv::Mat>&
    TrackImages()
    { return _track_mat_v; }
    
    const cv::Mat&
    TrackImage(size_t planeid)
    { return _track_mat_v[planeid]; }

    const std::vector<cv::Mat>&
    ShowerImages()
    { return _shower_mat_v; }
    
    const cv::Mat&
    ShowerImage(size_t planeid)
    { return _shower_mat_v[planeid]; }
    
  private:
    std::string _adc_producer;
    std::string _track_producer;
    std::string _shower_producer;
    
    LArbysImageMaker _LArbysImageMaker;

    // The images
    std::vector<cv::Mat> _adc_mat_v;
    std::vector<larocv::ImageMeta> _adc_meta_v;
    std::vector<cv::Mat> _track_mat_v;
    std::vector<cv::Mat> _shower_mat_v;
    
  };

  class LArbysImageExtractProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageExtractProcessFactory() { ProcessFactory::get().add_factory("LArbysImageExtract",this); }
    /// dtor
    ~LArbysImageExtractProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImageExtract(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

