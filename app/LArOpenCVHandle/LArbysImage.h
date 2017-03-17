#ifndef __LARBYSIMAGE_H__
#define __LARBYSIMAGE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArOpenCV/Core/ImageManager.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterManager.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterViewer.h"
#include "DataFormat/Image2D.h"
#include "PreProcessor.h"
#include "LArbysImageMaker.h"
#include "LArbysRecoHolder.h"
#include "ImageMod/ImageModUtils.h"

namespace larcv {

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

    const PreProcessor& PProcessor() const { return _PreProcessor; }
    const LArbysImageMaker& LArbysImgMaker() const { return _LArbysImageMaker; }
    const LArbysRecoHolder& LArbysHolder() const { return _reco_holder; }
    
  protected:

    const std::vector<larcv::Image2D>& get_image2d(IOManager& mgr, std::string producer);

    bool Reconstruct(const std::vector<larcv::Image2D>& adc_image_v,
		     const std::vector<larcv::Image2D>& track_image_v,
		     const std::vector<larcv::Image2D>& shower_image_v);

    bool StoreParticles(IOManager& iom,
			larocv::ImageClusterManager& mgr,
			const std::vector<larcv::Image2D>& adcimg_v);
    
    TTree* _tree;
    
    ::larocv::ImageClusterManager _alg_mgr;
    ::larocv::ImageManager _adc_img_mgr;
    ::larocv::ImageManager _track_img_mgr;
    ::larocv::ImageManager _shower_img_mgr;

    bool _debug;
    bool _preprocess;
    bool _write_reco;
    
    std::string _output_module_name;
    size_t _output_module_offset;
    
    std::vector<float> _plane_weights;
    std::string _adc_producer;
    std::string _roi_producer;
    std::string _track_producer;
    std::string _shower_producer;
    std::string _output_producer;
    ::larocv::AlgorithmID_t _output_cluster_alg_id;

    double _process_count;
    double _process_time_image_extraction;
    double _process_time_analyze;
    double _process_time_cluster_storage;

    void Report() const;
    
    PreProcessor _PreProcessor;
    LArbysImageMaker _LArbysImageMaker;
    LArbysRecoHolder _reco_holder;
    
    std::vector<larcv::Image2D> _empty_image_v;
  };

  /**
     \class larcv::LAr
bysImageFactory
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

