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
 * The class is a concrete implementation of a LArCV::ProcesBase.
 * It coordinates reading in different DLCosmicTag information
 *  formats that information in opencv::Mat objects which get passed
 *  to a LArOpenCV clustering algorithm manager implementing a sequence of
 *  of 2D-based vertex reconstruction.
 *
 * One key is that the DLCosmicTag inputs are partitioned into 3D clusters.
 * These clusters have 3D hits along with the associated 2D pixels within each plane
 *   found by collecting pixels around the projection of the 3D points into the planes.
 * We attempt to reduce false vertices by only searching for vertices on each of the planes
 *  that occur on the same 3D cluster.
 * To do this, we apply the vertex algorithms on several cropped images per event, 
 *  with one cropped image set displaying the 2D projections of charge for only one cluster at time.
 * 
 * Configuration parameter sets
 * 
 * - LArbysImageMaker
 * - DLCosmicTagUtil
 * - PreProcessor
 */

// LArOpenCV
#include "LArOpenCV/Core/ImageManager.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterManager.h"

// LArCV
#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/EventROI.h"

// LArCV/app
#include "LArOpenCVHandle/PreProcessor.h"
#include "LArOpenCVHandle/LArbysImageMaker.h"

// this module
#include "DLCosmicTagUtil.h"

namespace larcv {

  class DLCosmicTagVertexReco : public ProcessBase {

  public:
    
    /// Default constructor
    DLCosmicTagVertexReco(const std::string name="DLCosmicTagVertexReco");
    
    /// Default destructor
    virtual ~DLCosmicTagVertexReco(){}

    /// concrete implementatin of ProcessBase::configure
    void configure(const PSet&)  override;

    /// concrete implementatin of ProcessBase::initialize
    void initialize() override;

    /// concrete implementatin of ProcessBase::process    
    bool process(IOManager& mgr) override;

    /// concrete implementatin of ProcessBase::finalize
    void finalize() override;

    /// get the 2D LArOpenCV-based algorithm manager
    const larocv::ImageClusterManager& Manager() const { return _alg_mgr; }

    //const PreProcessor&     PProcessor()     const { return _PreProcessor; }
    //const DLCosmicTagVertexRecoMaker& LArbysImgMaker() const { return _DLCosmicTagVertexRecoMaker; }
    
  protected:

    /// get an image out of the provided LArCV IOManager (provided by the larcv::Processor for each entry)
    const std::vector<larcv::Image2D>& get_image2d(IOManager& mgr, std::string producer);

    /// get the current (run,subrun,event,entry) index
    void get_rsee(IOManager& mgr, std::string producer,
		  int& run, int& subrun, int& event, int& entry);
            
    /// run the LArOpenCV algorithm sequence
    bool reconstructClusters( IOManager& mgr );
    
    /// store the results of the vertex algorithm for one cluster
    bool storeClusterParticles(IOManager& iom,
                               size_t cluster_index,
                               const std::vector<Image2D>& adc_image_v,                               
                               size_t& pidx);

    /// The manager that organizes the LArOpenCV algorithms
    larocv::ImageClusterManager _alg_mgr;

    /// algorithm names and IDs (for retrieval from algo manager)
    std::string m_vertex_algo_name;
    std::string m_par_algo_name;
    std::string m_3D_algo_name;
    larocv::AlgorithmID_t m_vertex_algo_id;
    larocv::AlgorithmID_t m_par_algo_id;
    larocv::AlgorithmID_t m_3D_algo_id;
    size_t m_vertex_algo_vertex_offset; //?
    size_t m_par_algo_par_offset; //?
    
    
    /// Containers for cv::Mat+larocv::ImageMeta for each type
    larocv::ImageManager _adc_img_mgr;
    larocv::ImageManager _track_img_mgr;
    larocv::ImageManager _shower_img_mgr;
    larocv::ImageManager _thrumu_img_mgr;
    larocv::ImageManager _stopmu_img_mgr;
    larocv::ImageManager _chstat_img_mgr;
    
    /// clear the imagemanager containers
    void clearImageManagers();

    /// flag to mark the state: true=filled false=unfilled
    bool fCVMatImagesMade;    

    ::fcllite::PSet _image_cluster_cfg;

    /* bool _debug; */
    bool fRunPreprocessor;
    /* bool _write_reco; */

    /* bool _mask_thrumu_pixels; */
    /* bool _mask_stopmu_pixels; */
    
    /* std::string _output_module_name; */
    /* size_t _output_module_offset; */

    // IO configuration params

    // input producer (i.e. tree) names
    std::string m_rse_producer;
    std::string m_adc_producer;
    std::string m_chstatus_producer;
    
    /* std::string _roi_producer; */
    /* std::string _track_producer; */
    /* std::string _shower_producer; */

    /* ProductType_t _tags_datatype; */

    /* std::string _thrumu_producer; */
    /* std::string _stopmu_producer; */
    /* std::string _channel_producer; */
    std::string m_output_producer;

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

    // LAOpenCV image preprocessor (for ADC, shower, track)
    PreProcessor     m_preprocessor;

    // LArbysImageMaker methods and members
    // ------------------------------------

    /// utility class for converting image2d -> cv::Mat
    LArbysImageMaker m_imagemaker;
    
    // DLCosmicTagUtil methods and members
    // ------------------------------------

    /// interface to larlite::pixelmask data containing DL net output
    DLCosmicTagUtil m_dlcosmictag_input;

    /// wrapper around number of clusters for current entry
    size_t numClusters() { return m_dlcosmictag_input.numClusters(); };
    
    /* std::vector<larcv::Image2D> _empty_image_v; */
    /* std::vector<larcv::Image2D> _thrumu_image_v; */
    /* std::vector<larcv::Image2D> _stopmu_image_v; */

    /* ROI _current_roi; */

    /* bool _store_shower_image; */
    /* std::string _shower_pixel_prod; */

    /// convert larlite dlcosmictag pixelmask into larcv::image2d then into cv::Mat and store
    DLCosmicTagClusterImageCrops_t
      prepareClusterCVMatFromDLCosmicTagUtil( size_t icluster,
                                              const std::vector<larcv::Image2D>& adc_wholeview_v,
                                              const std::vector<larcv::Image2D>& chstatus_wholeview_v,
                                              bool& status );
    
    

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

