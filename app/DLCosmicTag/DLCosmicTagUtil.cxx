#include "DLCosmicTagUtil.h"

namespace larcv {

  /**
   * static instance used for logger access in static functions
   */
  DLCosmicTagUtil* DLCosmicTagUtil::_g_logger_instance = nullptr;
  
  /**
   * Default constructor
   *
   */  
  DLCosmicTagUtil::DLCosmicTagUtil()
    : fConfigured(false),
      _entry(-1),
      _io(nullptr),
      m_larflowcluster_v(nullptr),
      fEntryLoaded(false),
      _dloutput_entry(-1)
  {}

  /**
   * Default destructor
   *
   */  
  DLCosmicTagUtil::~DLCosmicTagUtil() {
    if (_io ) {
      _io->close();
      delete _io;
    }
  }
    
  /**
   *  Configure class parameters
   *
   * @param[in] pset Parameter set for class.
   */  
  void DLCosmicTagUtil::Configure( PSet& pset ) {

    /// Configurable parameters
    
    /// InputFilename: larlite file containing larflowcluster and pixelmask objects
    _larlite_input_filename_v = pset.get< std::vector<std::string> >("InputFilenames");

    /// IntimeLArFlowClusterProducer: name of tree containing intime larflow cluster objects
    _intime_larflowcluster_producername = pset.get<std::string>("IntimeLArFlowClusterProducer");
    
    /// IntimePixelMaskProducer: name of trees containing intime pixel mask objects. should be in plane order.
    _intime_pixelmask_producername_v    = pset.get<std::vector<std::string> >("IntimePlanePixelMaskProducers");
    if ( _intime_pixelmask_producername_v.size()==0 ) {
      LARCV_ERROR() << "Empty producer list for IntimePlanePixelMaskProducers" << std::endl;
    }

    /// SSNetShowerProducer: name of tree containing SSNet shower (whole image) pixel mask
    _ssnet_shower_producername = pset.get<std::string>("SSNetShowerProducer");

    /// SSNetTrackProducer: name of tree containing SSNet track (whole image) pixel mask
    _ssnet_track_producername = pset.get<std::string>("SSNetTrackProducer");

    /// SSNetEndptProducer: name of tree containing SSNet endpt (whole image) pixel mask
    _ssnet_endpt_producername = pset.get<std::string>("SSNetEndptProducer");
    
    /// InfillProducer: name of tree containing Infill (wholeimage) pixel mask
    _infill_producername = pset.get<std::string>("InfillProducer");
    
    /// Load the storage manager
    _io = new larlite::storage_manager( larlite::storage_manager::kREAD );
    for ( auto& input_filename : _larlite_input_filename_v ) 
      _io->add_in_filename( input_filename );
    _io->open();
    fConfigured = true;
  }

  /**
   *  Loads the entry from the input larlite tree (via larlite::storage_manager object)
   * 
   * @param[in] entry Entry number in TChain (or TTree).
   */    
  void DLCosmicTagUtil::goto_entry( size_t entry ) {

    // check if the class has been properly configured
    if ( !fConfigured ) {
      LARCV_ERROR() << "The class has not been configured yet!" << std::endl;
    }
    
    // load entry
    io().go_to( entry );
    _entry = io().get_index();

    // get data objects for the entry
    // ------------------------------

    // larflow clusters
    m_larflowcluster_v = (larlite::event_larflowcluster*)get_data( larlite::data::kLArFlowCluster,
								  _intime_larflowcluster_producername );

    // pixelmask for the clusters
    m_pixelmask_vv.resize( _intime_pixelmask_producername_v.size(), nullptr );
    for ( size_t iproducer=0; iproducer<_intime_pixelmask_producername_v.size(); iproducer++ ) {
      m_pixelmask_vv[iproducer] = (larlite::event_pixelmask*) get_data( larlite::data::kPixelMask, 
									_intime_pixelmask_producername_v[iproducer] );
    }
    
    // make sure the larflow and pixelmask products are as expected
    size_t nclusters = m_larflowcluster_v->size();    
    if ( nclusters!=m_pixelmask_vv.front()->size() ) {
      LARCV_ERROR() << "Number of larflowclusters and pixelmasks in the entry (" << _entry << ") do not match."
		    << " clusters=" << nclusters
		    << " pixelmasks=" << m_pixelmask_vv.front()->size()
		    << std::endl;
    }

    for ( auto const& pevent_pixelmask : m_pixelmask_vv ) {
      if ( pevent_pixelmask->size()!=nclusters ) {
	LARCV_ERROR() << "Pixelmasks in entry don't agree across planes. " << nclusters
		      << " vs " << pevent_pixelmask->size() << std::endl;
      }
    }

    LARCV_DEBUG() << "Loaded " << m_larflowcluster_v->size() << " larflowclusters and pixelmasks" << std::endl;

    // load ssnet and infill pixelmasks
    m_dloutput_masks_v.resize(4,nullptr);
    m_dloutput_masks_v[kShower] = (larlite::event_pixelmask*) get_data(larlite::data::kPixelMask,
                                                                        _ssnet_shower_producername);
    m_dloutput_masks_v[kTrack]  = (larlite::event_pixelmask*) get_data(larlite::data::kPixelMask,
                                                                       _ssnet_track_producername);
    m_dloutput_masks_v[kEndpt]  = (larlite::event_pixelmask*) get_data(larlite::data::kPixelMask,
                                                                       _ssnet_endpt_producername);
    m_dloutput_masks_v[kInfill] = (larlite::event_pixelmask*) get_data(larlite::data::kPixelMask,
                                                                       _infill_producername);
    
    fEntryLoaded = true;
  }

  /**
   *  Get number of clusters in the entry
   * 
   * @return number of clusters in the entry
   */      
  int DLCosmicTagUtil::getNumClusters() const {
    // check proper condition of class
    if ( !fEntryLoaded )
      LARCV_ERROR() << "Entry has not been loaded yet." << std::endl;
    
    return m_larflowcluster_v->size();
  }
  
  /**
   *  Wrapper around larlite::storage_manager::get_data
   * 
   * @param[in] data_type larlite::data enumerate ID.
   * @param[in] producername Name of tree to get object from.
   */      
  larlite::event_base* DLCosmicTagUtil::get_data( larlite::data::DataType_t data_type, std::string producername ) {
    return _io->get_data( data_type, producername );
  }

  /**
   * make the image crops for a given cluster (specified by the index)
   *
   * @param[in] cluster_index Index of larflowcluster in the current entry.
   * @param[in] adc_wholeview_v ADC images of the whole event view, one for each plane.
   *
   * @return Struct containing image2d crops for the cluster
   */
  DLCosmicTagClusterImageCrops_t DLCosmicTagUtil::makeClusterCrops( int cluster_index,
                                                                    const std::vector<larcv::Image2D>& adc_wholeview_v ) {
    // check if entry is loaded
    if ( !fEntryLoaded ) {
      LARCV_ERROR() << "An entry has not been loaded" << std::endl;
    }
    
    // get the larflow cluster and the pixel masks for each plane
    //const larlite::larflowcluster& lfcluster = m_larflowcluster_v->at(cluster_index);
    std::vector<const larlite::pixelmask*> mask_v(adc_wholeview_v.size(),nullptr);
    for ( size_t p=0; p<adc_wholeview_v.size(); p++ ) {
      mask_v[p] = &(m_pixelmask_vv.at(p)->at( cluster_index ));
    }

    DLCosmicTagClusterImageCrops_t output;
    output.cluster_index = cluster_index;

    // define the meta for the crops
    LARCV_DEBUG() << "Define the crop meta" << std::endl;
    for ( size_t p=0; p<adc_wholeview_v.size(); p++ ) {
      auto const& adcmeta = adc_wholeview_v.at(p).meta();
      if ( mask_v[p]->len()>0 ) {
        LARCV_DEBUG() << "defining crop for " << mask_v[p]->len() << " points" << std::endl;
        larcv::ImageMeta cropmeta = DLCosmicTagUtil::metaFromPixelMask( *mask_v[p], p, 50, &adc_wholeview_v.at(p).meta() );
        LARCV_DEBUG() << "cropmeta: " << cropmeta.dump();
        output.cropmeta_v.emplace_back( std::move(cropmeta) );
      }
      else {
        larcv::ImageMeta cropmeta( 0, 0, 0, 0, adcmeta.min_x(), adcmeta.max_y(), adcmeta.plane() );
        LARCV_DEBUG() << "define meta for empty pixelmask: " << cropmeta.dump();        
        output.cropmeta_v.emplace_back( std::move(cropmeta) );
      }
    }

    // make the whole-view DL Output images, if this is a new entry
    if ( getCurrentEntry()!=_dloutput_entry ) {
      m_dloutput_wholeview_vv.clear();
      m_dloutput_wholeview_vv.resize( kNumDLOutputs );
      for (int ioutput=0; ioutput<kNumDLOutputs; ioutput++ ) {
        auto& m_dloutput_wholeview_v = m_dloutput_wholeview_vv.at(ioutput);
        auto const& dloutput_mask_v    = *(m_dloutput_masks_v[ioutput]);
        LARCV_DEBUG() << "Making wholeview image for DLOutput Type=" << ioutput << std::endl;
        for ( size_t p=0; p<adc_wholeview_v.size(); p++ ) {
          larcv::Image2D dloutimg = DLCosmicTagUtil::image2dFromPixelMask( dloutput_mask_v.at(p), adc_wholeview_v.at(p).meta() );
          m_dloutput_wholeview_v.emplace_back( std::move(dloutimg) );
        }
      }
    }

    // crop from the DL output whole view objects
    for ( size_t p=0; p<adc_wholeview_v.size(); p++ ) {
      
      auto const& cropmeta = output.cropmeta_v.at(p);

      LARCV_DEBUG() << "crop for mask" << std::endl;
      larcv::Image2D maskcrop   = DLCosmicTagUtil::image2dFromPixelMask( *mask_v[p], cropmeta );
      output.clustermask_v.emplace_back( std::move(maskcrop) );

      LARCV_DEBUG() << "shower crop: " << cropmeta.dump();
      larcv::Image2D showercrop = m_dloutput_wholeview_vv[kShower].at(p).crop( cropmeta );
      output.ssnet_shower_v.emplace_back( std::move(showercrop) );

      LARCV_DEBUG() << "track crop: " << cropmeta.dump();      
      larcv::Image2D trackcrop = m_dloutput_wholeview_vv[kTrack].at(p).crop( cropmeta );
      output.ssnet_track_v.emplace_back( std::move(trackcrop) );

      LARCV_DEBUG() << "end-point crop: " << cropmeta.dump();
      larcv::Image2D endptcrop = m_dloutput_wholeview_vv[kEndpt].at(p).crop( cropmeta );
      output.ssnet_endpt_v.emplace_back( std::move(endptcrop) );

      LARCV_DEBUG() << "infill crop: " << cropmeta.dump();
      larcv::Image2D infillcrop = m_dloutput_wholeview_vv[kInfill].at(p).crop( cropmeta );
      output.infill_v.emplace_back( std::move(infillcrop) );
    }

    // pixel mask crop
    // std::vector<larcv::Image2D> clustercrop_v = makeIntimeCroppedImage( adc_wholeview_v, 50 );
    // for ( auto& clustercrop : clustercrop_v )
    //   output.clustermask_v.emplace_back( std::move(clustercrop) );


    return output;
  }

  /**
   *  Provide a cropped image around all intime pixelmasks
   *  It is expected that entry has been loaded using @see DLCosmicTagUtil#go_to_entry
   * 
   * @param[in] adc_wholeview_v Input images with ADC values to crop from.
   * @param[in] padding Number of pixels to add around mask. Default is 10.
   * @return vector of cropped images; entries align with input images.
   */  
  std::vector< larcv::Image2D >
  DLCosmicTagUtil::makeCombinedIntimeCroppedImage( const std::vector<larcv::Image2D>& adc_wholeview_v,
                                                   const int padding ) const {

    // check proper condition of class
    if ( !fEntryLoaded )
      LARCV_ERROR() << "Entry has not been loaded yet." << std::endl;

    // the numbers of planes should match
    if ( adc_wholeview_v.size()!=m_pixelmask_vv.size() ) {
      LARCV_ERROR() << "Number of input ADC plane images (" << adc_wholeview_v.size() << ") "
		    << "does not match with the number of pixelmask planes (" << m_pixelmask_vv.size() << ")"
		    << std::endl;
    }

    // the output vector
    std::vector< larcv::Image2D > img_v;

    //return empty vector if no clusters
    if ( m_larflowcluster_v->size()==0 )
      return img_v;
    
    // we get the bounding boxes from each of the pixelmasks and form a axis-aligned union bounding box
    // the bounding boxes are represented by the image2d imagemeta class
    std::vector< larcv::ImageMeta > meta_v;

    // setup the first meta(s)
    size_t plane = 0;
    for ( auto const& pevent_pixelmask : m_pixelmask_vv ) {
      auto const& mask = pevent_pixelmask->front();
      larcv::ImageMeta meta
        = DLCosmicTagUtil::metaFromPixelMask( mask, plane, padding, &adc_wholeview_v.at(plane).meta() );
      meta_v.emplace_back( std::move(meta) );
      plane++;
    }

    // now loop through the rest of the masks, updating the imagemeta
    for ( plane=0; plane<m_pixelmask_vv.size(); plane++ ) {
      size_t nclusters = m_pixelmask_vv[plane]->size();
      for ( size_t icluster=1; icluster<nclusters; icluster++ ) {
	auto const& mask = (*m_pixelmask_vv[plane])[icluster];
        if ( mask.len()==0 ) continue;
	larcv::ImageMeta cluster_meta
          = DLCosmicTagUtil::metaFromPixelMask( mask, plane, padding, &adc_wholeview_v.at(plane).meta() );
	larcv::ImageMeta& union_meta  = meta_v.at(plane);
	union_meta = union_meta.inclusive( cluster_meta );
      }
    }

    // now that we have union metas for each plane, crop out the regions
    // if for some reason we extend outside the range, we make an intersection box
    for ( plane=0; plane<m_pixelmask_vv.size(); plane++ ) {
      larcv::ImageMeta& unionmeta = meta_v.at(plane);
      const larcv::ImageMeta& adcmeta = adc_wholeview_v.at(plane).meta();
      if ( !adcmeta.contains( unionmeta ) ) {
	unionmeta = unionmeta.overlap( adcmeta );
      }

      // make crop
      larcv::Image2D crop = adc_wholeview_v.at(plane).crop( unionmeta );
      img_v.emplace_back( std::move(crop) );
    }
    
    return img_v;
  }

  // /**
  //  *  Provide a cropped image around one intime pixelmask
  //  *  It is expected that entry has been loaded using @see DLCosmicTagUtil#goto_entry
  //  * 
  //  * @param[in] cluster_index Index of cluster to provide crop+mask
  //  * @param[in] adc_wholeview_v Input images with ADC values to crop from.
  //  * @param[in] padding Number of pixels to add around mask. Default is 10.
  //  * @return vector of cropped images; entries align with input images.
  //  */  
  // std::vector< larcv::Image2D >
  // DLCosmicTagUtil::makeIntimeCroppedImage( const int cluster_index,
  //                                          const std::vector<larcv::Image2D>& adc_wholeview_v,
  //                                          const int padding ) const {

  //   // check proper condition of class
  //   if ( !fEntryLoaded )
  //     LARCV_ERROR() << "Entry has not been loaded yet." << std::endl;

  //   // the numbers of planes should match
  //   if ( adc_wholeview_v.size()!=m_pixelmask_vv.size() ) {
  //     LARCV_ERROR() << "Number of input ADC plane images (" << adc_wholeview_v.size() << ") "
  //       	    << "does not match with the number of pixelmask planes (" << m_pixelmask_vv.size() << ")"
  //       	    << std::endl;
  //   }

  //   // the output vector
  //   std::vector< larcv::Image2D > img_v;

  //   //return empty vector if no clusters
  //   if ( m_larflowcluster_v->size()<=cluster_index ) {
  //     LARCV_ERROR() << "Index of cluster requested (" << cluster_index << ") out of bounds" << std::endl;
  //   }
    
  //   // we get the bounding boxes from each of the pixelmasks and form a axis-aligned union bounding box
  //   // the bounding boxes are represented by the image2d imagemeta class
  //   std::vector< larcv::ImageMeta > meta_v;

  //   for ( size_t plane=0; plane<adc_wholeview_v.size(); plane++ ) {
  //     auto const& mask = pevent_pixelmask->at(cluster_index).at(plane);
  //     larcv::ImageMeta meta
  //       = DLCosmicTagUtil::metaFromPixelMask( mask, plane, padding, &adc_wholeview_v.at(plane).meta() );
  //     meta_v.emplace_back( std::move(meta) );
  //   }

  //   // now loop through the rest of the masks, updating the imagemeta
  //   for ( plane=0; plane<m_pixelmask_vv.size(); plane++ ) {
  //     size_t nclusters = m_pixelmask_vv[plane]->size();
  //     for ( size_t icluster=1; icluster<nclusters; icluster++ ) {
  //       auto const& mask = (*m_pixelmask_vv[plane])[icluster];
  //       if ( mask.len()==0 ) continue;
  //       larcv::ImageMeta cluster_meta
  //         = DLCosmicTagUtil::metaFromPixelMask( mask, plane, padding, &adc_wholeview_v.at(plane).meta() );
  //       larcv::ImageMeta& union_meta  = meta_v.at(plane);
  //       union_meta = union_meta.inclusive( cluster_meta );
  //     }
  //   }

  //   // now that we have union metas for each plane, crop out the regions
  //   // if for some reason we extend outside the range, we make an intersection box
  //   for ( plane=0; plane<m_pixelmask_vv.size(); plane++ ) {
  //     larcv::ImageMeta& unionmeta = meta_v.at(plane);
  //     const larcv::ImageMeta& adcmeta = adc_wholeview_v.at(plane).meta();
  //     if ( !adcmeta.contains( unionmeta ) ) {
  //       unionmeta = unionmeta.overlap( adcmeta );
  //     }

  //     // make crop
  //     larcv::Image2D crop = adc_wholeview_v.at(plane).crop( unionmeta );
  //     img_v.emplace_back( std::move(crop) );
  //   }
    
  //   return img_v;
  // }
  
  /**
   *  Mask the input image with the pixel masks from the entry.
   *  It is expected that entry has been loaded using @see DLCosmicTagUtil#go_to_entry
   *
   * @param[in] mask Input vector of ADC images
   * @return vector of masked ADC images
   */  
  std::vector< larcv::Image2D > DLCosmicTagUtil::makeCosmicMaskedImage(  const std::vector<larcv::Image2D>& adc_view_v ) const {

    // check proper condition of class
    if ( !fEntryLoaded )
      LARCV_ERROR() << "Entry has not been loaded yet." << std::endl;

    // the numbers of planes should match
    if ( adc_view_v.size()!=m_pixelmask_vv.size() ) {
      LARCV_ERROR() << "Number of input ADC plane images (" << adc_view_v.size() << ") "
		    << "does not match with the number of pixelmask planes (" << m_pixelmask_vv.size() << ")"
		    << std::endl;
    }
    
    // create blank copies of the input images
    std::vector< larcv::Image2D > img_v;
    for ( auto const& img : adc_view_v ) {
      larcv::Image2D blank( img.meta() );
      blank.paint(0.0);
      img_v.emplace_back( std::move(blank) );
    }

    // if we have no clusters, we provide an empty mask
    if ( m_larflowcluster_v->size()==0 )
      return img_v;

    // now mask. we transfer only those pixels from the pixel masks.
    size_t nplanes = adc_view_v.size();
    for ( size_t iplane=0; iplane<nplanes; iplane++ ) {

      // get the input image and the output masked image
      const larcv::Image2D&   adcimg  = adc_view_v.at(iplane);      
      const larcv::ImageMeta& adcmeta = adcimg.meta();

      larcv::Image2D& masked = img_v.at(iplane);

      // loop over the pixelmask objects for this plane
      auto const& pevent_pixelmask = m_pixelmask_vv.at(iplane);      
      for ( auto const& mask : *pevent_pixelmask ) {

	// for each point, check if inside input image meta
	// if so, copy adc value to output
	LARCV_DEBUG() << "Masking image on plane=" << iplane << " with " << mask.len() << " pixels" << std::endl;
	for ( int ipt=0; ipt<mask.len(); ipt++ ) {
	  std::vector<float> xy = mask.point(ipt);
	  if ( ! adcmeta.contains( xy[0], xy[1] ) )
	    continue;

	  int adccol = adcmeta.col( xy[0] );
	  int adcrow = adcmeta.row( xy[1] );
	  masked.set_pixel( adcrow, adccol, adcimg.pixel( adcrow, adccol ) );
	}
	
      }
    }
    
    return img_v;
  }

  /**
   *  Make images where cluster instances are marked. Same indices are used across planes for 2D matching.
   *  The ID used is simply the index of the cluster as ordered in the event container.
   *  It is expected that entry has been loaded using @see DLCosmicTagUtil#go_to_entry
   *
   * @param[in] adc_view_v Input vector of ADC images
   * @return vector of tagged images
   */  
  std::vector< larcv::Image2D > DLCosmicTagUtil::makeClusterTaggedImages(  const std::vector<larcv::Image2D>& adc_view_v ) const {

    // check proper condition of class
    if ( !fEntryLoaded )
      LARCV_ERROR() << "Entry has not been loaded yet." << std::endl;

    // the numbers of planes should match
    if ( adc_view_v.size()!=m_pixelmask_vv.size() ) {
      LARCV_ERROR() << "Number of input ADC plane images (" << adc_view_v.size() << ") "
		    << "does not match with the number of pixelmask planes (" << m_pixelmask_vv.size() << ")"
		    << std::endl;
    }
    
    // create blank copies of the input images
    std::vector< larcv::Image2D > img_v;
    for ( auto const& img : adc_view_v ) {
      larcv::Image2D blank( img.meta() );
      blank.paint(0.0);
      img_v.emplace_back( std::move(blank) );
    }

    // if we have no clusters, we provide an empty mask
    if ( m_larflowcluster_v->size()==0 )
      return img_v;

    // now mask+tag. we tag only those pixels from the pixel masks.
    size_t nplanes = adc_view_v.size();
    for ( size_t iplane=0; iplane<nplanes; iplane++ ) {

      // get the input image and the output masked image
      const larcv::Image2D& adcimg = adc_view_v.at(iplane);      
      const larcv::ImageMeta& adcmeta = adcimg.meta();

      larcv::Image2D& masked = img_v.at(iplane);

      // loop over the pixelmask objects for this plane
      auto const& pevent_pixelmask = m_pixelmask_vv.at(iplane);
      
      for ( int icluster=0; icluster<(int)pevent_pixelmask->size(); icluster++ ) {
	auto const& mask = pevent_pixelmask->at(icluster);

	// for each point, check if inside input image meta
	// if so, copy adc value to output
	for ( int ipt=0; ipt<mask.len(); ipt++ ) {
	  std::vector<float> xy = mask.point(ipt);
	  if ( ! adcmeta.contains( xy[0], xy[1] ) )
	    continue;

	  int adccol = adcmeta.col( xy[0] );
	  int adcrow = adcmeta.row( xy[1] );
	  masked.set_pixel( adcrow, adccol, icluster );
	}
	
      }
    }
    
    return img_v;    
  }
  
  
  /**
   *  utility to make larcv::imagemeta from larlite::pixelmask. static function.
   *
   * @param[in] mask Input PixelMask.
   * @param[in] planeid (optional) PlaneID to be assigned to output imagemeta. Default is 0.
   * @param[in] padding (optional) Pixels to pad around mask's bounding box. Default is 50.
   * @param[in] bounding_meta (optional) If provided, pixelmask is bounded by this meta. Default nullptr.
   * @return ImageMeta representing bounding box around pixels in pixelmask.
   */
  larcv::ImageMeta DLCosmicTagUtil::metaFromPixelMask( const larlite::pixelmask& mask,
                                                       unsigned int planeid,
                                                       const int padding,
                                                       const larcv::ImageMeta* bounding_meta) {
    
    std::vector<float> bbox;
    // if ( bounding_meta )
    //   bbox = mask.as_vector_bbox( bounding_meta->min_x(), bounding_meta->min_y(),
    //                               bounding_meta->max_x(), bounding_meta->max_y() );
    // else
    bbox = mask.as_vector_bbox();

    if ( !bounding_meta ) {
      bbox[0] -= padding;
      bbox[1] -= padding;
      bbox[2] += padding;
      bbox[3] += padding;
    }
    else {
      bbox[0] -= padding*bounding_meta->pixel_width();
      bbox[1] -= padding*bounding_meta->pixel_height();
      bbox[2] += padding*bounding_meta->pixel_width();
      bbox[3] += padding*bounding_meta->pixel_height();
    }
      
    DLCOSMICTAGUTIL_INFO() << "bbox: " << bbox[0] << "," << bbox[1] << "," << bbox[2] << "," << bbox[3]  << std::endl;

    int nrows = mask.rows();
    int ncols = mask.cols();

    if ( bounding_meta ) {
      if ( bbox[0]<bounding_meta->min_x() ) bbox[0] = bounding_meta->min_x();
      if ( bbox[1]<bounding_meta->min_y() ) bbox[1] = bounding_meta->min_y();
      if ( bbox[2]>bounding_meta->max_x() ) bbox[2] = bounding_meta->max_x();
      if ( bbox[3]>bounding_meta->max_y() ) bbox[3] = bounding_meta->max_y();
    }

    // pixel coordinate height and widths
    float width  = bbox[2]-bbox[0];
    float height = bbox[3]-bbox[1];
    if ( bounding_meta ) {
      nrows = height/bounding_meta->pixel_height();
      ncols = width/bounding_meta->pixel_width();

      if ( (nrows*bounding_meta->pixel_height() - height > 1.0e-3 )
           || (ncols*bounding_meta->pixel_width() - width)>1.0e-3 ) {
        DLCOSMICTAGUTIL_WARNING()
          << "crop width or height is not an integer multiple of pixel width or height"
          << std::endl;
      }
    }
    
    // origin for larcv1 is (minx, maxy)
    larcv::ImageMeta meta( width, height, nrows, ncols,
			   bbox[0], bbox[3], (larcv::PlaneID_t)planeid );
    
    return meta;
  }

  /**
   *  utility to make larcv::Image2D from larlite::pixelmask. static function.
   *
   * @param[in] mask Input PixelMask.
   * @param[in] outputmeta Meta that defines output image within which we will embed the data from the pixelmask
   * @return Image2D containing the pixelmask data.
   */
  larcv::Image2D DLCosmicTagUtil::image2dFromPixelMask( const larlite::pixelmask& mask, const larcv::ImageMeta& outputmeta ) {
    
    larcv::Image2D output( outputmeta );
    output.paint(0);

    // for each point, check if inside input image meta
    // if so, copy adc value to output
    int ncontains = 0;
    //DLCOSMICTAGUTIL_INFO() << "fill " << mask.len() << " points into outputmeta=" << outputmeta.dump();
    for ( int ipt=0; ipt<mask.len(); ipt++ ) {
      std::vector<float> xy = mask.point(ipt);
      //std::cout << "(" << xy[0] << "," << xy[1] << ": " << xy[2] << ") ";
      if ( ! outputmeta.contains( xy[0], xy[1] ) )
        continue;

      int adccol = outputmeta.col( xy[0] );
      int adcrow = outputmeta.row( xy[1] );
      if ( mask.dim_per_point()==2 ) {
        output.set_pixel( adcrow, adccol, 1 );
      }
      else if ( mask.dim_per_point()>2 ) {
        //DLCOSMICTAGUTIL_INFO() << "(" << xy[0] << "," << xy[1] << ") " << xy[2] << std::endl;
        output.set_pixel( adcrow, adccol, xy.at(2) );
      }
      else
        DLCOSMICTAGUTIL_ERROR() << "larlite::pixelmask dims per point should be >=2" << std::endl;
      ncontains++;
    }
    DLCOSMICTAGUTIL_INFO() << "filled " << ncontains << " pixels"
                           << "into output image="
                           << outputmeta.dump();
    
    return output;
  }


  /**
   * access to static instance used for logger in static functions
   *
   */
  const larcv::logger& DLCosmicTagUtil::get_logger() {
    if ( _g_logger_instance==nullptr )
      _g_logger_instance = new DLCosmicTagUtil();
    return _g_logger_instance->logger();
  }

  /**
   * Get wholeview images we made for the DL Output images
   *
   * @param[in] 
   */
  std::vector< larcv::Image2D >& DLCosmicTagUtil::getWholeViewDLOutputImage( DLCosmicTagUtil::DLOutput_t dltype ) {
    if ( !fEntryLoaded )
      LARCV_ERROR() << "Entry has not been loaded yet." << std::endl;

    if ( m_dloutput_wholeview_vv.size()!=kNumDLOutputs ) {
      LARCV_ERROR() << "Have not created the whole-view DLoutputs yet. "
                    << "(have to call makeClusterCrops first)"
                    << std::endl;
    }

    return m_dloutput_wholeview_vv.at(dltype);
  }
  
}
