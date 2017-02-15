#ifndef __LARBYSIMAGE_H__
#define __LARBYSIMAGE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArOpenCV/Core/ImageManager.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterManager.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterViewer.h"
#include "DataFormat/user_info.h"
#include "PreProcessor.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class LArbysImage ... these comments are used to generate
     doxygen documentation!
  */
  class LArbysImage : public ProcessBase {

  public:
    
    /// Default constructor
    LArbysImage(const std::string name="LArbysImage");
    
    /// Default destructor
    ~LArbysImage(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    const ::larocv::ImageClusterManager& Manager() const { return _alg_mgr; }

    const PreProcessor& PProcessor() const { return _pre_processor; }
    
  protected:

    void store_clusters(IOManager& mgr);
    void extract_image(IOManager& mgr);

    TTree* _tree;
    ::larlite::event_user* _eui;
    
    ::larocv::ImageClusterManager _alg_mgr;
    ::larocv::ImageManager _adc_img_mgr;
    ::larocv::ImageManager _track_img_mgr;
    ::larocv::ImageManager _shower_img_mgr;

    bool   _debug;
    double _charge_to_gray_scale;
    double _charge_min;
    double _charge_max;
    bool _preprocess;
    std::vector<float> _plane_weights;
    std::string _adc_producer;
    std::string _track_producer;
    std::string _shower_producer;
    std::string _output_producer;
    ::larocv::AlgorithmID_t _output_cluster_alg_id;
    double _process_count;
    double _process_time_image_extraction;
    double _process_time_analyze;
    double _process_time_cluster_storage;
    void Report() const;
    
    PreProcessor _pre_processor;
  };

  /**
     \class larcv::LArbysImageFactory
     \brief A concrete factory class for larcv::LArbysImage
  */
  class LArbysImageProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageProcessFactory() { ProcessFactory::get().add_factory("LArbysImage",this); }
    /// dtor
    ~LArbysImageProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImage(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

