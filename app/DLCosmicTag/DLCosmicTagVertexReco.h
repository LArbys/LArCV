#ifndef __DLCOSMICTAG_VERTEX_RECO_H__
#define __DLCOSMICTAG_VERTEX_RECO_H__

/**
 * 
 * \class DLCosmicTagVertexReco
 *
 * \defgroup DLCosmicTagVertexReco
 *
 * \brief Class responsible for running LArOpenCV vertex reconstruction
 *
 * A concrete implementation of a LArCV::Processor.
 * It coordinates reading in different DLCosmicTag information
 *  formats that information in opencv::Mat objects which get passed
 *  to a LArOpenCV clustering algorithm manager implementing 2D-based
 *  vertex reconstruction.
 *
 * One key is that the DLCosmicTag inputs are partitioned into 3D clusters.
 * We attempt to reduce false vertices by only searching for vertices on each of the planes
 *  that occur on the same 3D cluster.
 * To do this, we apply the vertex algorithms on several cropped images per event, 
 *  with one cropped image set displaying the 2D projections of charge for only one cluster at time.
 * 
 */

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArOpenCV/Core/ImageManager.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterManager.h"
#include "DataFormat/Image2D.h"
//#include "PreProcessor.h"
#include "DataFormat/EventROI.h"

#include "DLCosmicTagUtil.h"

namespace larcv {

  class DLCosmicTagVertexReco : public ProcessBase {

  public:
    
    /// Default constructor
    DLCosmicTagVertexReco(const std::string name="DLCosmicTagVertexReco");
    
    /// Default destructor
    virtual ~DLCosmicTagVertexReco(){}

    void configure(const PSet&)  override;

    void initialize() override;

    bool process(IOManager& mgr) override;

    void finalize() override;

    const larocv::ImageClusterManager& Manager() const { return _alg_mgr; }

    //const PreProcessor&     PProcessor()     const { return _PreProcessor; }
    //const DLCosmicTagVertexRecoMaker& LArbysImgMaker() const { return _DLCosmicTagVertexRecoMaker; }
    
  protected:

    const std::vector<larcv::Image2D>& get_image2d(IOManager& mgr, std::string producer);

    void get_rsee(IOManager& mgr, std::string producer,
		  int& run, int& subrun, int& event, int& entry);
            
    
    bool Reconstruct(const std::vector<larcv::Image2D>& adc_image_v,
		     const std::vector<larcv::Image2D>& track_image_v,
		     const std::vector<larcv::Image2D>& shower_image_v,
		     const std::vector<larcv::Image2D>& thrumu_image_v,
		     const std::vector<larcv::Image2D>& stopmu_image_v,
		     const std::vector<larcv::Image2D>& chstat_image_v);

    void Process();
    
    /* bool StoreParticles(IOManager& iom, */
    /*     		const std::vector<larcv::Image2D>& adcimg_v, */
    /*     		size_t& pidx); */

    larocv::ImageClusterManager _alg_mgr;
    /* larocv::ImageManager _adc_img_mgr; */
    /* larocv::ImageManager _track_img_mgr; */
    /* larocv::ImageManager _shower_img_mgr; */
    /* larocv::ImageManager _thrumu_img_mgr; */
    /* larocv::ImageManager _stopmu_img_mgr; */
    /* larocv::ImageManager _chstat_img_mgr; */

    ::fcllite::PSet _image_cluster_cfg;

    /* bool _debug; */
    /* bool _preprocess; */
    /* bool _write_reco; */

    /* bool _mask_thrumu_pixels; */
    /* bool _mask_stopmu_pixels; */
    
    /* std::string _output_module_name; */
    /* size_t _output_module_offset; */
    
    std::string _rse_producer;
    /* std::string _adc_producer; */
    /* std::string _roi_producer; */
    /* std::string _track_producer; */
    /* std::string _shower_producer; */

    /* ProductType_t _tags_datatype; */

    /* std::string _thrumu_producer; */
    /* std::string _stopmu_producer; */
    /* std::string _channel_producer; */
    /* std::string _output_producer; */

    /* std::string _vertex_algo_name; */
    /* std::string _par_algo_name; */
    /* std::string _3D_algo_name; */
    
    /* larocv::AlgorithmID_t _vertex_algo_id; */
    /* larocv::AlgorithmID_t _par_algo_id; */
    /* larocv::AlgorithmID_t _3D_algo_id; */
    
    /* size_t _vertex_algo_vertex_offset; */
    /* size_t _par_algo_par_offset; */
    
    /* double _process_count; */
    /* double _process_time_image_extraction; */
    /* double _process_time_analyze; */
    /* double _process_time_cluster_storage; */

    /* bool _union_roi; */
    
    /* void Report() const; */
    
    /* PreProcessor     _PreProcessor; */
    DLCosmicTagUtil m_dlcosmictag_input;
    
    /* std::vector<larcv::Image2D> _empty_image_v; */
    /* std::vector<larcv::Image2D> _thrumu_image_v; */
    /* std::vector<larcv::Image2D> _stopmu_image_v; */

    /* ROI _current_roi; */

    /* bool _store_shower_image; */
    /* std::string _shower_pixel_prod; */

  };

  /**
     \class larcv::DLCosmicTagVertexRecoFactory
     \brief A concrete factory class for larcv::DLCosmicTagVertexReco
  */
  class DLCosmicTagVertexRecoProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    DLCosmicTagVertexRecoProcessFactory() { ProcessFactory::get().add_factory("DLCosmicTagVertexReco",this); }
    /// dtor
    ~DLCosmicTagVertexRecoProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new DLCosmicTagVertexReco(instance_name); }
  };

}

#endif

