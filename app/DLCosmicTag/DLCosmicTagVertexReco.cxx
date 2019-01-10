#ifndef __DLCOSMICTAG_VERTEX_RECO_CXX__
#define __DLCOSMICTAG_VERTEX_RECO_CXX__

#include <sstream>

// ROOT
#include "TCanvas.h"
#include "TStyle.h"
#include "TPad.h"
#include "TExec.h"
#include "TPad.h"
#include "TMarker.h"

// larlite
#include  "LArUtil/LArProperties.h"
#include  "LArUtil/Geometry.h"

// LArOpenCV
#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"
#include "LArOpenCV/ImageCluster/AlgoData/ParticleCluster.h"
#include "LArOpenCV/ImageCluster/AlgoData/InfoCollection.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"

// LArCV
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/ROI.h"
#include "ROOTUtil/ROOTUtil.h"

// DLCosmicTag module
#include "DLCosmicTagVertexReco.h"

namespace larcv {

  static DLCosmicTagVertexRecoProcessFactory __global_DLCosmicTagVertexRecoProcessFactory__;

  /**
   * constructor
   *
   */
  DLCosmicTagVertexReco::DLCosmicTagVertexReco(const std::string name)
    : ProcessBase(name)
  {
  }
  
  /**
   * configure the class
   *
   * @param[in] cfg PSet class provided by larcv::Processor
   *
   */
  void DLCosmicTagVertexReco::configure(const PSet& cfg)
  {
    LARCV_DEBUG() << "start configure" << std::endl;
    
    // parameters for this class
    m_rse_producer         = cfg.get<std::string>("RSEImageProducer");
    m_adc_producer         = cfg.get<std::string>("ADCImageProducer");
    m_chstatus_producer    = cfg.get<std::string>("ChStatusImageProducer");
    m_output_producer      = cfg.get<std::string>("OutputImageProducer");
    fVisualize             = cfg.get<bool>("Visualize",false);
    
    // parameters for DLCosmicTagUtil (handles reading in larlite::pixelmask obejcts)
    auto dlcosmictag_cfg = cfg.get<larcv::PSet>("DLCosmicTagUtil");
    m_dlcosmictag_input.Configure( dlcosmictag_cfg );
    m_dlcosmictag_input.set_verbosity( (larcv::msg::Level_t)dlcosmictag_cfg.get<int>("Verbosity",0) );
    
    // parameters for LArbysImageMaker (handles conversion of image2d to cv::Mat)
    auto imagemaker_cfg = cfg.get<larcv::PSet>("LArbysImageMaker");
    m_imagemaker.Configure( imagemaker_cfg );
    
    // optional: setup the preprocessor
    fRunPreprocessor = cfg.get<bool>("PreProcess");
    if (fRunPreprocessor) {
      LARCV_INFO() << "Preprocessing image" << std::endl;
      auto preprocessor_cfg = cfg.get<larcv::PSet>("PreProcessor");
      m_preprocessor.Configure(preprocessor_cfg);
    }

    // parameters for vertex algo
    LARCV_DEBUG() << "Make Algo Configuration for Manager: " << _alg_mgr.Name() << std::endl;
    auto lcv_image_cluster_cfg = cfg.get<larcv::PSet>(_alg_mgr.Name());

    auto image_cluster_cfg = ::fcllite::PSet(_alg_mgr.Name(),lcv_image_cluster_cfg.data_string());
    m_vertex_algo_name    = cfg.get<std::string>("VertexAlgoName");
    m_par_algo_name       = cfg.get<std::string>("ParticleAlgoName");
    m_3D_algo_name        = cfg.get<std::string>("3DAlgoName");
    
    // configure the algo manager
    _alg_mgr.Configure( image_cluster_cfg.get_pset(_alg_mgr.Name()) );
    
    // get the IDs needed to grab output data for each algo
    const auto& data_man = _alg_mgr.DataManager();
    
    // vertex algo
    m_vertex_algo_id = larocv::kINVALID_ALGO_ID;
    if (!m_vertex_algo_name.empty()) {
      m_vertex_algo_id = data_man.ID(m_vertex_algo_name);
      if (m_vertex_algo_id == larocv::kINVALID_ALGO_ID)
	throw larbys("Specified invalid vertex algorithm");
    }

    // particle clustering algo
    m_par_algo_id = larocv::kINVALID_ALGO_ID;
    if (!m_par_algo_name.empty()) {
      m_par_algo_id = data_man.ID(m_par_algo_name);
      if (m_par_algo_id == larocv::kINVALID_ALGO_ID) {
	throw larbys("Specified invalid particle algorithm");
      }
    }

    // 3d consistency algo
    m_3D_algo_id = larocv::kINVALID_ALGO_ID;
    if (!m_3D_algo_name.empty()) {
      m_3D_algo_id = data_man.ID(m_3D_algo_name);
      if (m_3D_algo_id == larocv::kINVALID_ALGO_ID) {
	throw larbys("Specified invalid 3D info algorithm");
      }
    }
    
    // I don't know what these do
    m_vertex_algo_vertex_offset = cfg.get<size_t>("VertexAlgoVertexOffset",0);
    m_par_algo_par_offset       = cfg.get<size_t>("ParticleAlgoParticleOffset",0);
    
    // _store_shower_image = cfg.get<bool>("StoreShowerImage",false);
    // if(_store_shower_image) {
    //   _shower_pixel_prod = cfg.get<std::string>("ShowerPixelProducer");
    //   assert(!_shower_pixel_prod.empty());
    // }

    if ( !fVisualize )
      LARCV_INFO() << "Visualizing vertex reco output." << std::endl;
    
    std::cout << "FIN" << std::endl;

    return;
  }

  /**
   * initialize: does nothing
   */
  void DLCosmicTagVertexReco::initialize()
  {
  }

  // ==========================================================
  // Clearing methods

  /** 
   * clear laropencv Image containers
   *
   */
  void DLCosmicTagVertexReco::clearImageManagers() {
    _adc_img_mgr.clear();
    _track_img_mgr.clear();
    _shower_img_mgr.clear();
    _thrumu_img_mgr.clear();
    _stopmu_img_mgr.clear();
    _chstat_img_mgr.clear();
    fCVMatImagesMade = false;
  }
  
  // ==========================================================  

  /**
   * get the run, subrun, event, and entry number for the given IOManager entry
   *
   * @param[in] mgr IOManager containing larcv objects. Assumed to be loaded with the right entry.
   * @param[in] producer Name of tree to use to grab RSE. Usually the ADC image tree.
   * @param[inout] run Run number.
   * @param[inout] subrun Subrun number.
   * @param[inout] event Event number.
   * @param[inout] entry Entry number.
   */
  void DLCosmicTagVertexReco::get_rsee(IOManager& mgr,std::string producer,int& run, int& subrun, int& event, int& entry) {
    
    auto ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,producer));
    
    size_t local_run    = ev_image->run();
    size_t local_subrun = ev_image->subrun();
    size_t local_event  = ev_image->event();
    size_t local_entry  = mgr.current_entry();
    
    if (local_run == kINVALID_SIZE)  {
      LARCV_CRITICAL() << "Invalid run number @ entry " << local_entry << " from " << producer << " producer" << std::endl;
      throw larbys();
    }
    
    if (local_subrun == kINVALID_SIZE)  {
      LARCV_CRITICAL() << "Invalid subrun number @ entry " << local_entry << " from " << producer << " producer" << std::endl;
      throw larbys();
    }
    
    if (local_event == kINVALID_SIZE)  {
      LARCV_CRITICAL() << "Invalid event number @ entry " << local_entry << " from " << producer << " producer" << std::endl;
      throw larbys();
    }
    
    if (local_entry == kINVALID_SIZE)  {
      LARCV_CRITICAL() << "Invalid entry number @ entry "  << local_entry << " from " << producer << " producer" << std::endl;
      throw larbys();
    }
    
    run    = (int) local_run;
    subrun = (int) local_subrun;
    event  = (int) local_event;
    entry  = (int) local_entry;
    
  }

  /**
   * process the entry
   *
   * the RSEE is gathered from the IOManager.
   * this is used to load the corresponding entry for the larlite interface (m_dlcosmictag).
   * for each cluster, we make larcv crops, convert them to cv::Mat, pass them to the 
   *   laropencv vertexing algorithm, and store the vertex candidate information for each cluster.
   * we use the entry to sync the different larcv and larlite files. Thus, we strongly assume
   *   entry alignment.
   *
   * @param[in] mgr LArCV data interface. Provided by the Processor.
   *
   * @return Bool is used to filter event or not (if Processor configuration enables filtering)
   *
   */
  bool DLCosmicTagVertexReco::process(IOManager& mgr)
  {
    LARCV_DEBUG() << "Process Entry " << mgr.current_entry() << std::endl;
    

    bool status = true;
    
    // get the run, subrun, event, entry
    int run, subrun, event, entry;
    get_rsee(mgr,m_rse_producer,run,subrun,event,entry);
    
    // pass it to the algo manager
    _alg_mgr.SetRSEE(run,subrun,event,entry);
    
    // set the entry for the larlite data
    m_dlcosmictag_input.goto_entry(entry);
    
    // find candidate vertices (store result in IOManager)
    status = reconstructClusters( mgr );
        
    LARCV_DEBUG() << "return " << status << std::endl;
    LARCV_DEBUG() << std::endl;
    return status;
  } // end process
  
  /**
   * reconstruct: generate cv::Mat crops for each cluster, pass into laropencv vertex reco
   *
   * @param[in] mgr LArCV IOManager, with current entry already assumed to be loaded
   * @return true if one or more vertices are found
   *
   */
  bool DLCosmicTagVertexReco::reconstructClusters( IOManager& mgr ) 
  {
    
    LARCV_DEBUG() << "start: " << numClusters() << " clusters in this event" << std::endl;
    
    size_t numclusters = numClusters();

    if (numclusters==0)
      return false;
    
    // retrieve larcv images
    larcv::EventImage2D* ev_adcwire = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, m_adc_producer );
    larcv::EventImage2D* ev_chstat  = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, m_chstatus_producer );
    
    bool status = true;
    size_t vertices_found  = 0;
    size_t particles_found = 0;
    for ( size_t icluster=0; icluster<numclusters; icluster++ ) {

      size_t past_nvertices  = vertices_found;

      // clear the vertex algorithm for each cluster
      _alg_mgr.ClearData();
      
      // load laropencv crops into member ImageManagers
      larcv::DLCosmicTagClusterImageCrops_t crops
        = prepareClusterCVMatFromDLCosmicTagUtil( icluster,
                                                  ev_adcwire->Image2DArray(),
                                                  ev_chstat->Image2DArray(),
                                                  status );
      
      // insert images into algorithm manager
      
      // ADC values
      for ( size_t plane=0; plane < _adc_img_mgr.size(); plane++ ) {
        auto       & img  = _adc_img_mgr.img_at(plane);
        const auto & meta = _adc_img_mgr.meta_at(plane);
        if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
        LARCV_DEBUG() << "... add adc (SetID=" << (int)larocv::ImageSetID_t::kImageSetWire << ") "
                      << "@plane="<<plane<<std::endl;
        _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetWire);
      }
            
      // ssnet: track
      for (size_t plane = 0; plane < _track_img_mgr.size(); ++plane) {
        auto       & img  = _track_img_mgr.img_at(plane);
        const auto & meta = _track_img_mgr.meta_at(plane);
        if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
        LARCV_DEBUG() << "... add track (SetID=" << (int)larocv::ImageSetID_t::kImageSetTrack << ") "
                      << "@plane="<<plane<<std::endl;
        _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetTrack);
      }

      // ssnet: shower
      for (size_t plane = 0; plane < _shower_img_mgr.size(); ++plane) {
        auto       & img  = _shower_img_mgr.img_at(plane);
        const auto & meta = _shower_img_mgr.meta_at(plane);
        if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
        LARCV_DEBUG() << "... add shower (SetID=" << (int)larocv::ImageSetID_t::kImageSetShower << ") "
                      << "@plane="<<plane<<std::endl;
        _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetShower);
      }      
      
      // chstatus
      for (size_t plane = 0; plane < _chstat_img_mgr.size(); ++plane) {
        auto       & img  = _chstat_img_mgr.img_at(plane);
        const auto & meta = _chstat_img_mgr.meta_at(plane);
        if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
        LARCV_DEBUG() << "... add chstat (SetID=" << (int)larocv::ImageSetID_t::kImageSetChStatus << ") "
                      << "@plane="<<plane<<std::endl;
        _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetChStatus);
      }
      

      if (fRunPreprocessor) {
        // give a single plane @ a time to pre processor
        auto& adc_img_v= _alg_mgr.InputImages(larocv::ImageSetID_t::kImageSetWire);
        auto& trk_img_v= _alg_mgr.InputImages(larocv::ImageSetID_t::kImageSetTrack);
        auto& shr_img_v= _alg_mgr.InputImages(larocv::ImageSetID_t::kImageSetShower);
        auto nplanes = adc_img_v.size();
        for(size_t plane_id=0;plane_id<nplanes;++plane_id) {
          LARCV_DEBUG() << "Preprocess image set @ "<< " plane " << plane_id << std::endl;
          if (!m_preprocessor.PreProcess(adc_img_v[plane_id],trk_img_v[plane_id],shr_img_v[plane_id])) {
            LARCV_CRITICAL() << "... could not be preprocessed, abort!" << std::endl;
            throw larbys();
          }
        }
      }//run preprocessor
      
      // run reconstruction
      _alg_mgr.Process();      
      
      // gather candidate vertex output
      storeClusterParticles( mgr, icluster, crops.clustermask_v, vertices_found, particles_found );

      // if visualize
      if  ( fVisualize ) {
        visualizeAlgoOutput( mgr, icluster, crops, past_nvertices );
      }
      
    }//end of cluster loop
    
    LARCV_DEBUG() << "End of vertex reco: reconstructed ID'd a total of " << vertices_found << std::endl;
    
    return true;
  }

  /**
   * converts and stores larcv image crops from DLCosmicTagUtil for one cluster
   *
   * uses DLCosmicTagUtil to generate image2d crops for each cluster.
   * uses LArOpenCVHandle/LArbysImageMaker to convert image2d to cv::Mat
   *
   * @param[in] icluster Cluster index.
   * @param[in] adc_wholeview_v vector of whole-view ADC images (one for each plane)
   * @param[in] chstatus_wholeview_v vector of chstatus-marked whole-view images (one for each plane)
   * @param[inout] status true if everything OK.
   * @return[in] struct containing 2D crops of various DL input images around cluster pixels
   */
    larcv::DLCosmicTagClusterImageCrops_t
      DLCosmicTagVertexReco::prepareClusterCVMatFromDLCosmicTagUtil( size_t icluster,
                                                                     const std::vector<larcv::Image2D>& adc_wholeview_v,
                                                                     const std::vector<larcv::Image2D>& chstatus_wholeview_v,
                                                                     bool& status ) {
    
    // empty containers for cv::Mat
    clearImageManagers();

    // should be parameter
    std::vector<float> adc_threshold_v(3,10.0);
    std::vector<float> ssnet_threshold_v(3,0.3);    
    
    // generator crop for cluster
    larcv::DLCosmicTagClusterImageCrops_t crops
      = m_dlcosmictag_input.makeClusterCrops( icluster, adc_wholeview_v );


    // we generate the images shower/track images we need
    //  which are overlays of the ADC onto ssnet-value masks
    std::vector<larcv::Image2D> adc_showerimg_v;
    std::vector<larcv::Image2D> adc_trackimg_v;
    makeSSNetMaskedADCImages( crops, adc_threshold_v, ssnet_threshold_v,
                              adc_showerimg_v, adc_trackimg_v );
    
    // fill the containers

    // ADC image: required
    std::vector< std::tuple<cv::Mat,larocv::ImageMeta> > cvdata_adc
      = m_imagemaker.ExtractImage( crops.clustermask_v );
    for ( auto& tup : cvdata_adc )
      _adc_img_mgr.emplace_back( std::move( std::get<0>(tup) ), std::move(std::get<1>(tup)) );
    
    // shower image: required
    std::vector< std::tuple<cv::Mat,larocv::ImageMeta> > cvdata_shower
      = m_imagemaker.ExtractImage( adc_showerimg_v );
    for ( auto& tup : cvdata_shower )
      _shower_img_mgr.emplace_back( std::move( std::get<0>(tup) ), std::move(std::get<1>(tup)) );
    
    // track image: required
    std::vector< std::tuple<cv::Mat,larocv::ImageMeta> > cvdata_track
      = m_imagemaker.ExtractImage( adc_trackimg_v );
    for ( auto& tup : cvdata_track )
      _track_img_mgr.emplace_back( std::move( std::get<0>(tup) ), std::move(std::get<1>(tup)) );
    
    // (we have no cosmic image as we've masked them out already)
    
    // chstatus (larcv)
    std::vector<larcv::Image2D> chstatus_crop_v;
    for ( size_t iplane=0; iplane<chstatus_wholeview_v.size(); iplane++ ) {
      auto const& cropmeta = crops.cropmeta_v.at(iplane);
      auto const& chstatus_img = chstatus_wholeview_v.at(iplane);
      chstatus_crop_v.push_back( chstatus_img.crop( cropmeta ) );
    }
    std::vector< std::tuple<cv::Mat,larocv::ImageMeta> > cvdata_chstatus
      = m_imagemaker.ExtractImage( chstatus_crop_v );
    for ( auto& tup : cvdata_chstatus )
      _chstat_img_mgr.emplace_back( std::move( std::get<0>(tup) ), std::move(std::get<1>(tup)) );
    
    fCVMatImagesMade = true;
    status = true;
    
    return crops;
  }

  /** 
   * mask ADC images using SSNet information for cluster crop images
   *
   */
  void DLCosmicTagVertexReco::makeSSNetMaskedADCImages( const DLCosmicTagClusterImageCrops_t& crops,
                                                        const std::vector<float> adc_threshold_v,
                                                        const std::vector<float> ssnet_background_threshold_v,
                                                        std::vector<larcv::Image2D>& adc_showerimg_v,
                                                        std::vector<larcv::Image2D>& adc_trackimg_v )
  {

    adc_showerimg_v.clear();
    adc_trackimg_v.clear();
    for ( size_t iplane=0; iplane<crops.clustermask_v.size(); iplane++ ) {
      
      auto const& adc    = crops.clustermask_v.at(iplane);
      auto const& shower = crops.ssnet_shower_v.at(iplane);
      auto const& track  = crops.ssnet_track_v.at(iplane);
      auto const& endpt  = crops.ssnet_endpt_v.at(iplane);

      if ( shower.meta()!=track.meta() || shower.meta()!=adc.meta() ) {
        LARCV_ERROR() << "shower,track, and adc meta does not match!" << std::endl;
      }

      float ssnet_threshold = ssnet_background_threshold_v[iplane];
      float adc_threshold   = adc_threshold_v[iplane];
      
      larcv::Image2D adc_shower( adc.meta() );
      larcv::Image2D adc_track( adc.meta() );
      adc_shower.paint(0);
      adc_track.paint(0);
      
      for ( size_t irow=0; irow<adc.meta().rows(); irow++ ) {
        for ( size_t icol=0; icol<adc.meta().cols(); icol++ ) {
          
          if ( adc.pixel(irow,icol)<adc_threshold )
            continue;
          
          float tot = endpt.pixel(irow,icol)+shower.pixel(irow,icol)+track.pixel(irow,icol);
          if ( tot<ssnet_threshold )
            continue;
          
          if ( shower.pixel(irow,icol)>track.pixel(irow,icol) ) {
            adc_shower.set_pixel( irow, icol, adc.pixel(irow,icol) );
          }
          else {
            adc_track.set_pixel( irow, icol, adc.pixel(irow,icol));
          }
          
        }//end of col loop
      }//end of row loop

      adc_showerimg_v.emplace_back( std::move(adc_shower ) );
      adc_trackimg_v.emplace_back( std::move(adc_track) );

    }//end of plane loop
  }
    

  /**
   * retrieve the vertex results after applying to a cluster crop
   *
   * @param[inout] iom larcv IOManager which we will fill with results.
   * @param[in] cluster_index Index of cluster for this result
   * @param[in] adc_image_v Cropped ADC image in which we searched
   * @param[inout] nvertices_found Count increatemented for each vertex stored
   * @param[inout] nparticles_found Counter incremented for each particle stored
   *
   */
  bool DLCosmicTagVertexReco::storeClusterParticles(IOManager& iom,
                                                    size_t cluster_index,
                                                    const std::vector<Image2D>& adc_image_v,
                                                    size_t& nvertices_found,
                                                    size_t& nparticles_found) {
    
    LARCV_DEBUG() << "store results for cluster[" << cluster_index << "]" << std::endl;

    // nothing to be done
    if (m_vertex_algo_id == larocv::kINVALID_ALGO_ID) {
      LARCV_INFO() << "Nothing to be done..." << std::endl;
      return true;
    }

    // retrieve the cv::Mat crop used for the clusters
    const auto& adc_cvimg_orig_v = _alg_mgr.InputImages(larocv::ImageSetID_t::kImageSetWire);

    // why static? we going recursive?
    static std::vector<cv::Mat> adc_cvimg_v;
    adc_cvimg_v.clear();
    adc_cvimg_v.resize(3);

    // retrieve/create output event container in IOManager

    // particles attached to the vertex
    auto event_pgraph           = (EventPGraph*)  iom.get_data(kProductPGraph,  m_output_producer);

    // ? 2D contours around particles
    auto event_ctor_pixel       = (EventPixel2D*) iom.get_data(kProductPixel2D, m_output_producer + "_ctor");

    // pixelmask around particles
    auto event_img_pixel        = (EventPixel2D*) iom.get_data(kProductPixel2D, m_output_producer + "_img");

    // ?
    auto event_ctor_super_pixel = (EventPixel2D*) iom.get_data(kProductPixel2D, m_output_producer + "_super_ctor");

    // ?
    auto event_img_super_pixel  = (EventPixel2D*) iom.get_data(kProductPixel2D, m_output_producer + "_super_img");

    // get the container for the algo output products
    const auto& data_mgr = _alg_mgr.DataManager();

    // get the associations for the algo products
    const auto& ass_man  = data_mgr.AssManager();

    // get the list of candiate vertices
    const auto vtx3d_array = (larocv::data::Vertex3DArray*) data_mgr.Data( m_vertex_algo_id, m_vertex_algo_vertex_offset );
    const auto& vtx3d_v = vtx3d_array->as_vector();

    // get the list of particles attached to vertices
    const auto par_array = (larocv::data::ParticleArray*) data_mgr.Data( m_par_algo_id, m_par_algo_par_offset );
    const auto& par_v = par_array->as_vector();

    // get the 3D matching results
    const auto info_3D_array = (larocv::data::Info3DArray*) data_mgr.Data( m_3D_algo_id, 0 );
    const auto& info_3D_v = info_3D_array->as_vector();


    // number of candidate vertices
    auto n_reco_vtx = vtx3d_v.size();
    
    LARCV_DEBUG() << "For cluster[" << cluster_index << "] "
                  << "found " << n_reco_vtx << " candidate vertices" << std::endl;

    // loop and store
    for(size_t vtxid=0; vtxid< n_reco_vtx; ++vtxid) {
      
      const auto& vtx3d = vtx3d_v[vtxid];

      LARCV_DEBUG() << "cluster[" << cluster_index << "] vertex[" << vtxid << "] "
                    << "@ (x,y,z) : ("<<vtx3d.x<<","<<vtx3d.y<<","<<vtx3d.z<<")"<<std::endl;

      // make a copy of the cropped regions
      for(size_t plane=0; plane<3; ++plane)
        adc_cvimg_v[plane] = adc_cvimg_orig_v[plane].clone();

      // find associated particles to this vertex
      auto par_id_v = ass_man.GetManyAss(vtx3d,par_array->ID());
      if (par_id_v.empty()) {
        LARCV_DEBUG() << "No associated particles to vertex " << vtxid << ": not-saved" << std::endl;
        continue;
      }

      // retrieve super-particle cluster (gather shower fragments?) for each plane
      std::vector<const larocv::data::ParticleCluster*> super_pcluster_v(3,nullptr);
      for(size_t plane=0; plane<3; ++plane) {
        const auto super_par_array
          = (larocv::data::ParticleClusterArray*) data_mgr.Data( m_vertex_algo_id, 1+plane+3 );
        const auto& super_par_v = super_par_array->as_vector();
        auto super_ass_id_v = ass_man.GetManyAss(vtx3d,super_par_array->ID());
        if (super_ass_id_v.empty()) continue;
        assert (super_ass_id_v.size()==1);
        super_pcluster_v[plane] = &(super_par_v.at(super_ass_id_v.front()));
      }

      // form output larcv product: a particle graph object
      PGraph pgraph;
      for(const auto& par_id : par_id_v) {
	
        const auto& par = par_v.at(par_id);

        const auto info3d_id   = ass_man.GetOneAss(par,info_3D_array->ID());
        if (info3d_id == kINVALID_SIZE) throw larbys("Particle unassociated to 3D info collection");
        const auto& par_info3d = info_3D_v.at(info3d_id);
	
        // New ROI for this particle
        ROI proi;
	
        // Store the vertex
        proi.Position(vtx3d.x,vtx3d.y,vtx3d.z,kINVALID_DOUBLE);

        // Store the end point
        proi.EndPosition(par_info3d.overall_pca_end_pt.at(0),
        		 par_info3d.overall_pca_end_pt.at(1),
        		 par_info3d.overall_pca_end_pt.at(2),
        		 kINVALID_DOUBLE);
	
        // Store the type
        if      (par.type==larocv::data::ParticleType_t::kTrack)   proi.Shape(kShapeTrack);
        else if (par.type==larocv::data::ParticleType_t::kShower)  proi.Shape(kShapeShower);
	  
        // Push the ROI into the PGraph
        LARCV_DEBUG() << " @ pg array index " << nvertices_found << std::endl;

        // save the crop meta around this
        for(size_t plane=0; plane<3; ++plane) 
          proi.AppendBB(adc_image_v[plane].meta());

        // store
        pgraph.Emplace(std::move(proi),nparticles_found);
        nparticles_found++;

        // @ Each plane, store pixels and contour per matched particle
        for(size_t plane=0; plane<3; ++plane) {

          const auto& pmeta = adc_image_v[plane].meta();
	    
          std::vector<Pixel2D> super_pixel_v, super_ctor_v;
          std::vector<Pixel2D> pixel_v, ctor_v;

          const auto& pcluster = par._par_v[plane];
          const auto& pctor = pcluster._ctor;
	  
          const auto& img2d = adc_image_v.at(plane);
          const auto& cvimg = adc_cvimg_v.at(plane);
	  
          if(!pctor.empty()) {

            const auto& super_pcluster = super_pcluster_v[plane];
            if (!super_pcluster) {
              LARCV_CRITICAL() << "particle exists but super contour not present" << std::endl;
              throw larbys("die");
            }
            const auto& super_pctor = super_pcluster->_ctor;
            auto super_pctor_masked = larocv::MaskImage(cvimg,super_pctor,0,false);
            auto super_par_pixel_v  = larocv::FindNonZero(super_pctor_masked);

            auto pctor_masked = larocv::MaskImage(cvimg,pctor,0,false);
            auto par_pixel_v  = larocv::FindNonZero(pctor_masked);

            super_pixel_v.reserve(super_par_pixel_v.size());
            super_ctor_v.reserve(super_pctor.size());

            pixel_v.reserve(par_pixel_v.size());
            ctor_v.reserve(pctor.size());
	    
            // Store super particle Image2D pixels
            for (const auto& px : super_par_pixel_v) {
              auto col  = cvimg.cols - px.x - 1;
              auto row  = px.y;
              auto gray = img2d.pixel(col,row);
              super_pixel_v.emplace_back(col,row);
              super_pixel_v.back().Intensity(gray);
            }	    
	    
            // Store the particle Image2D pixels
            for (const auto& px : par_pixel_v) {
              auto col  = cvimg.cols - px.x - 1;
              auto row  = px.y;
              auto gray = img2d.pixel(col,row);
              pixel_v.emplace_back(col,row);
              pixel_v.back().Intensity(gray);
            }

            // Store super contour
            for(const auto& pt : super_pctor)  {
              auto col  = cvimg.cols - pt.x - 1;
              auto row  = pt.y;
              auto gray = 1.0;
              super_ctor_v.emplace_back(row,col);
              super_ctor_v.back().Intensity(gray);
            }

            // Store contour
            for(const auto& pt : pctor)  {
              auto col  = cvimg.cols - pt.x - 1;
              auto row  = pt.y;
              auto gray = 1.0;
              ctor_v.emplace_back(row,col);
              ctor_v.back().Intensity(gray);
            }
          } // empty particle contour on this plane
          
          Pixel2DCluster super_pixcluster(std::move(super_pixel_v));
          Pixel2DCluster super_pixctor(std::move(super_ctor_v));
          
          Pixel2DCluster pixcluster(std::move(pixel_v));
          Pixel2DCluster pixctor(std::move(ctor_v));
          
          event_img_super_pixel->Emplace(plane,std::move(super_pixcluster),pmeta);
          event_ctor_super_pixel->Emplace(plane,std::move(super_pixctor),pmeta);
          
          event_img_pixel->Emplace(plane,std::move(pixcluster),pmeta);
          event_ctor_pixel->Emplace(plane,std::move(pixctor),pmeta);

        } // end this plane
      } // end this particle

      // store the pgraph
      event_pgraph->Emplace(std::move(pgraph));

      // kept this vertex
      nvertices_found++;
      
    } // end vertex

    LARCV_DEBUG() << "Event pgraph cummulative size " << event_pgraph->PGraphArray().size() << std::endl;
    return true;
  }
  
  void DLCosmicTagVertexReco::finalize()
  {
    if ( has_ana_file() )  {
      _alg_mgr.Finalize(&(ana_file()));
    }
  }

  /**
   * visualize the different components of the algorithm
   *
   * displays a tcanvas and/or png
   *
   * @param[in] mgr IOManager provided by ProcessDriver that we just filled with reconstruct
   *
   */
  void DLCosmicTagVertexReco::visualizeAlgoOutput( larcv::IOManager& mgr,
                                                   const int cluster_index,
                                                   const larcv::DLCosmicTagClusterImageCrops_t& crops,
                                                   const size_t start_vertex_id ) {

    gStyle->SetOptStat(0);
    
    // get the trees
    
    // particles attached to the vertex
    auto event_pgraph           = (EventPGraph*)  mgr.get_data(kProductPGraph,  m_output_producer);

    // ? 2D contours around particles
    auto event_ctor_pixel       = (EventPixel2D*) mgr.get_data(kProductPixel2D, m_output_producer + "_ctor");

    // pixelmask around particles
    auto event_img_pixel        = (EventPixel2D*) mgr.get_data(kProductPixel2D, m_output_producer + "_img");

    // ?
    auto event_ctor_super_pixel = (EventPixel2D*) mgr.get_data(kProductPixel2D, m_output_producer + "_super_ctor");

    // ?
    auto event_img_super_pixel  = (EventPixel2D*) mgr.get_data(kProductPixel2D, m_output_producer + "_super_img");

    size_t nvertices = event_pgraph->PGraphArray().size();

    // first make the tcanvas
    TCanvas c("cdlcosmictagvertexreco", "DL Cosmic Tag Vertext Reco", 1400, 800 );
    c.Divide( 3, 2 ); // (3 planes, {ADC,SSNET})

    int run, subrun, event, entry;
    get_rsee(mgr,m_rse_producer,run,subrun,event,entry);

    // geo constant
    const float cm_per_tick = larutil::LArProperties::GetME()->DriftVelocity()*0.5;

    // adc images
    std::stringstream hadc_name;
    hadc_name << "hadc_entry" << entry << "_cluster" << cluster_index;
    std::vector< TH2D > hadc_v = larcv::as_th2d_v( crops.clustermask_v, hadc_name.str() );

    // ssnet images: we create an image of -log(shower) + log(track)
    std::vector< larcv::Image2D > ssnet;
    float tick_min_ymax = 1e4;
    float tick_max_ymin = 0;
    for ( size_t p=0; p<3; p++ ) {
      auto const& adc    = crops.clustermask_v[p];
      auto const& shower = crops.ssnet_shower_v[p];
      auto const& track  = crops.ssnet_track_v[p];
      larcv::Image2D ss( adc.meta() );
      ss.paint(0);
      for ( size_t irow=0; irow<adc.meta().rows(); irow++ ) {
        for ( size_t icol=0; icol<adc.meta().cols(); icol++ ) {
          if ( adc.pixel(irow,icol)<10 ) continue;
          float score = log( shower.pixel(irow,icol) ) - log( track.pixel(irow,icol) );
          ss.set_pixel( irow, icol, score );
        }
      }
      ssnet.emplace_back( std::move(ss) );
      if ( adc.meta().min_y() > tick_max_ymin )
        tick_max_ymin = adc.meta().min_y();
      if ( adc.meta().max_y() < tick_min_ymax )
        tick_min_ymax = adc.meta().max_y();
    } // end of plane loop for making ssnet image
    
    std::stringstream hssnet_name;
    hssnet_name << "hssnet_entry" << entry << "_cluster" << cluster_index;    
    std::vector< TH2D > hssnet_v = larcv::as_th2d_v( ssnet, hssnet_name.str() );

    // particle objects
    std::vector< std::vector<TGraph> > tcontours_per_vertex[3]; // array index for planes
    for ( size_t p=0; p<3; p++ )
      tcontours_per_vertex[p].resize( nvertices-start_vertex_id );
    std::vector< TMarker > tmarker_vertex[3];
    
    for ( size_t ivert=start_vertex_id; ivert<nvertices; ivert++ ) {
      const PGraph& pg = event_pgraph->at(ivert);
      
      const std::vector< larcv::ROI > proi = pg.ParticleArray();
      const std::vector< size_t > pixelcluster_index_v = pg.ClusterIndexArray();
      
      size_t ip=0;
      for ( auto const& idx_particle : pixelcluster_index_v ) {
        auto const& roi = proi[ip];
        ip++;
        
        if ( idx_particle==event_ctor_pixel->Pixel2DClusterArray(0).size() )
          LARCV_ERROR() << "number of particles in the event vertices "
                        << "greater than number stored in contour container" << std::endl;
        
        // get clusters for this particle
        for ( size_t p=0; p<3; p++ ) {
          auto const& particle_contour = event_ctor_pixel->Pixel2DClusterArray(p).at(idx_particle);
          TGraph tcont = larcv::as_contour_tgraph( particle_contour, true, &crops.clustermask_v[p].meta() );
          tcont.SetLineWidth(1);
          if ( roi.Shape()==kShapeShower ) {
            tcont.SetLineColor(kRed);
            tcont.SetMarkerColor(kRed);
          }
          else if (roi.Shape()==kShapeTrack ) {
            tcont.SetLineColor(kBlue);
            tcont.SetMarkerColor(kBlue);
          }
          tcontours_per_vertex[p][ivert-start_vertex_id].emplace_back( std::move(tcont) );
        }

        // make the vertex (projected into planes)
        Double_t pos[3] = { roi.X(), roi.Y(), roi.Z() };
        float tick = roi.X()/cm_per_tick + 3200;
        for ( size_t p=0; p<3; p++ ) {
          float wire = larutil::Geometry::GetME()->NearestWire( pos, p );
          TMarker vmark( wire, tick, 29 );
          vmark.SetMarkerColor(kOrange);
          vmark.SetMarkerSize(3);
          tmarker_vertex[p].emplace_back( std::move(vmark) );
        }
      }//loop over particles in vertex
    }// loop over vertices
    
    // draw ADC images
    larcv::setRedHeatColorPalette();
    //TExec pad1( "expad1", "larcv::setRedHeatColorPalette();");
    for ( size_t p=0; p<3; p++ ) {
      c.cd( p+1 );
      hadc_v[p].GetYaxis()->SetRangeUser( tick_max_ymin, tick_min_ymax );      
      hadc_v[p].Draw("colz");
      //pad1.Draw();
      hadc_v[p].Draw("colz same");
      for ( auto& vert : tcontours_per_vertex[p] ) {
        for ( auto& g : vert ) {
          g.Draw("L");
        }
      }
      for ( auto& vert : tmarker_vertex[p] )
        vert.Draw();
    }
    
    // draw SSNet image
    larcv::setBlueRedColorPalette();
    //TExec pad2( "expad2", "larcv::setBlueRedColorPalette();" );
    
    for ( size_t p=0; p<3; p++ ) {
      c.cd( p+4 );
      hssnet_v[p].GetYaxis()->SetRangeUser( tick_max_ymin, tick_min_ymax );
      hssnet_v[p].SetMaximum(6);
      hssnet_v[p].SetMinimum(-6);
      hssnet_v[p].Draw("colz");
      //pad2.Draw();
      hssnet_v[p].Draw("colz same");
      for ( auto& vert : tcontours_per_vertex[p] ) {
        for ( auto& g : vert ) {
          g.Draw("L");
        }
      }
      for ( auto& vert : tmarker_vertex[p] ) {
        vert.SetMarkerColor( kOrange+7 );
        vert.Draw();
      }
    }

    std::stringstream canvname;
    canvname << "vertex_run" << run << "_subrun" << subrun << "_event" << event
             << "_entry" << entry << "_cluster" << cluster_index << ".png";
    c.Draw();
    c.Update();
    c.SaveAs( canvname.str().c_str() );
    std::cout << "[Enter to continue]" << std::endl;
    //std::cin.get();
  }
  
}
#endif
