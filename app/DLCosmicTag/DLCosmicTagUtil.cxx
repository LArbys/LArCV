#include "DLCosmicTagUtil.h"

namespace larcv {

  /**
   * Default constructor
   *
   */  
  DLCosmicTagUtil::DLCosmicTagUtil()
    : _entry(-1),
      _io(nullptr),
      m_larflowcluster_v(nullptr),
      fEntryLoaded(false)
  {}
    
  /**
   *  Configures class
   *
   * @param[in] pset Parameter set for class.
   */  
  void DLCosmicTagUtil::Configure( PSet& pset ) {

    /// Configurable parameters
    
    /// InputFilename: larlite file containing larflowcluster and pixelmask objects
    _larlite_input_filename             = pset.get<std::string>("InputFilename");

    /// IntimePixelMaskProducer: name of trees containing intime pixel mask objects. should be in plane order.
    _intime_pixelmask_producername_v    = pset.get<std::vector<std::string> >("IntimePlanePixelMaskProducers");
    if ( _intime_pixelmask_producername_v.size()==0 ) {
      LARCV_ERROR() << "Empty producer list for IntimePlanePixelMaskProducers" << std::endl;
    }

    /// IntimeLArFlowClusterProducer: name of tree containing intime larflow cluster objects
    _intime_larflowcluster_producername = pset.get<std::string>("IntimeLArFlowClusterProducer");

    _io = new larlite::storage_manager( larlite::storage_manager::kREAD );
    
  }

  /**
   *  Loads the entry from the input larlite tree (via larlite::storage_manager object)
   * 
   * @param[in] entry Entry number in TChain (or TTree).
   */    
  void DLCosmicTagUtil::go_to_entry( size_t entry ) {

    // load entry
    io().go_to( entry );
    _entry = io().get_index();

    // get data objects for the entry
    m_larflowcluster_v = (larlite::event_larflowcluster*)get_data( larlite::data::kLArFlowCluster,
								  _intime_larflowcluster_producername );
    m_pixelmask_vv.resize( _intime_pixelmask_producername_v.size(), nullptr );
    for ( size_t iproducer=0; iproducer<_intime_pixelmask_producername_v.size(); iproducer++ ) {
      m_pixelmask_vv[iproducer] = (larlite::event_pixelmask*) get_data( larlite::data::kPixelMask, 
									_intime_pixelmask_producername_v[iproducer] );
    }
    
    // make sure the products are as expected
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
    fEntryLoaded = true;
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
   *  Provide a cropped image around the intime pixelmask
   *  It is expected that entry has been loaded using @see DLCosmicTagUtil#go_to_entry
   * 
   * @param[in] adc_wholeview_v Input images with ADC values to crop from.
   * @param[in] padding Number of pixels to add around mask. Default is 10.
   * @return vector of cropped images; entries align with input images.
   */  
  std::vector< larcv::Image2D > DLCosmicTagUtil::makeIntimeCroppedImage( const std::vector<larcv::Image2D>& adc_wholeview_v, const int padding ) const {

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
      larcv::ImageMeta meta = DLCosmicTagUtil::metaFromPixelMask( mask, plane );
      meta_v.emplace_back( std::move(meta) );
    }

    // now loop through the rest of the masks, updating the imagemeta
    for ( plane=0; plane<m_pixelmask_vv.size(); plane++ ) {
      size_t nclusters = m_pixelmask_vv[plane]->size();
      for ( size_t icluster=1; icluster<nclusters; icluster++ ) {
	auto const& mask = (*m_pixelmask_vv[plane])[icluster];
	larcv::ImageMeta cluster_meta = DLCosmicTagUtil::metaFromPixelMask( mask, plane );
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
      img_v.emplace_back( std::move(img) );
    }

    // if we have no clusters, we provide an empty mask
    if ( m_larflowcluster_v->size()==0 )
      return img_v;

    // now mask. we transfer only those pixels from the pixel masks.
    size_t nplanes = adc_view_v.size();
    for ( size_t iplane=0; iplane<nplanes; iplane++ ) {

      // get the input image and the output masked image
      const larcv::Image2D& adcimg = adc_view_v.at(iplane);      
      const larcv::ImageMeta& adcmeta = adcimg.meta();

      larcv::Image2D& masked = img_v.at(iplane);

      // loop over the pixelmask objects for this plane
      auto const& pevent_pixelmask = m_pixelmask_vv.at(iplane);      
      for ( auto const& mask : *pevent_pixelmask ) {

	// for each point, check if inside input image meta
	// if so, copy adc value to output
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
      img_v.emplace_back( std::move(img) );
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
   *  utility to make larcv::imagemeta from larlite::pixelmask
   *
   * @param[in] mask Input PixelMask.
   * @param[in] plane PlaneID to be assigned to output imagemeta.
   * @return ImageMeta representing bounding box around pixels in pixelmask.
   */
  larcv::ImageMeta DLCosmicTagUtil::metaFromPixelMask( const larlite::pixelmask& mask, unsigned int planeid ) {
    
    std::vector<float> bbox = mask.as_vector_bbox();
    // pixel coordinate height and widths
    float width  = bbox[2]-bbox[0];
    float height = bbox[3]-bbox[1];
    // origin for larcv1 is (minx, maxy)
    larcv::ImageMeta meta( width, height, mask.rows(), mask.cols(),
			   bbox[0], bbox[3], (larcv::PlaneID_t)planeid );
    
    return meta;
  }
  
}
