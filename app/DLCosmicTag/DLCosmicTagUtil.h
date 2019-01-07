#ifndef __DLCosmicTagUtil_h__
#define __DLCosmicTagUtil_h__

/**
 * @file DLCosmicTagUtil.h
 *
 * @brief Class to handle larlite DLCosmicTag Input
 *
 * @ingroup DLCosmicTag
 *
 * @author Taritree Wongjirad
 * Contact: twongj01@tufts.edu
 *
 */

#include <vector>

// LArCV
#include "Base/larcv_base.h"
#include "Base/PSet.h"
#include "DataFormat/Image2D.h"


// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/pixelmask.h"
#include "DataFormat/larflowcluster.h"

#define DLCOSMICTAGUTIL_ERROR()   DLCosmicTagUtil::get_logger().send( ::larcv::msg::kERROR,   __FUNCTION__, __LINE__ )
#define DLCOSMICTAGUTIL_WARNING() DLCosmicTagUtil::get_logger().send( ::larcv::msg::kWARNING, __FUNCTION__, __LINE__ )
#define DLCOSMICTAGUTIL_INFO()    DLCosmicTagUtil::get_logger().send( ::larcv::msg::kINFO,    __FUNCTION__, __LINE__ )
#define DLCOSMICTAGUTIL_DEBUG()    DLCosmicTagUtil::get_logger().send(::larcv::msg::kDEBUG,   __FUNCTION__, __LINE__ )

namespace larcv {

  /**
   * struct containing larcv::Image2D crops in each plane for a given larflowcluster
   */
  struct DLCosmicTagClusterImageCrops_t {
    int cluster_index;                              ///< larflowcluster index
    std::vector< larcv::ImageMeta > cropmeta_v;     ///< meta that define the crops
    std::vector< larcv::Image2D >   clustermask_v;  ///< ADC values for cluster
    std::vector< larcv::Image2D >   ssnet_shower_v; ///< SSNet shower
    std::vector< larcv::Image2D >   ssnet_track_v;  ///< SSNet track
    std::vector< larcv::Image2D >   ssnet_endpt_v;  ///< SSNet endpt
    std::vector< larcv::Image2D >   infill_v;       ///< Infill
  };
  
  /**
   * 
   * Class to handle larlite DLCosmicTag Input
   *
   * Responsible for loading larlite data for DLCosmicTagVertexReco.
   * This includes converting larlite objects into Image2D crops.
   *
   * For each larlite::larflowcluster (made of 3D hits), we generate image 
   *  crops for each plane around the image region the 3D hits project into.
   *  The List of images:
   *   - ssnet track scores
   *   - ssnet shower scores
   *   - ssnet endpoint scores
   *   - infill scores
   *   - intime-cluster pixelmask
   * 
   * Example Configuration pset:
   *
   * DLCosmicUtil: {
   *  InputFilenames: ["flashmatchfilled-larlite-Run000001-SubRun006867.root","dlcosmicstitched-larlite-Run000001-SubRun006867.root"]
   *    IntimeLArFlowClusterProducer: "intimeflashmatched"
   *    IntimePlanePixelMaskProducers: ["intimefilledp0","intimefilledp1","intimefilledp2"]
   *    SSNetShowerPixelMaskProducer: "shower"
   *    SSNetTrackPixelMaskProducer:  "track"
   *    SSNetEndptPixelMaskProducer:  "endpt"
   *    InfillPixelMaskProducer:      "infill"
   *  }     
   *
   */  
  class DLCosmicTagUtil : public larcv::larcv_base {
  public:

    typedef enum { kShower=0, kTrack, kEndpt, kInfill, kNumDLOutputs } DLOutput_t;
    
    DLCosmicTagUtil();
    virtual ~DLCosmicTagUtil();

    // configure the class
    void Configure( PSet& pset );

    // go to an entry in the larlite file
    void goto_entry( size_t entry );

    // exposes storage_manager::get_data
    larlite::event_base* get_data( larlite::data::DataType_t data_type, std::string producername );

    /// Get number of larflowclusters in the entry
    int numClusters() const;
    
    /// Make cluster crops for a given larflowcluster
    DLCosmicTagClusterImageCrops_t makeClusterCrops( int cluster_index,
                                                     const std::vector<larcv::Image2D>& adc_wholeview_v );
    
    /// use all the intime cluster objects to form a crop within the wholeview image
    std::vector< larcv::Image2D >
      makeCombinedIntimeCroppedImage( const std::vector<larcv::Image2D>& adc_wholeview_v,
                                      const int padding=50 ) const;

    /* /// use one of the intime cluster objects to form a crop within the wholeview image */
    /* std::vector< larcv::Image2D > makeIntimeCroppedImage( const int cluster_index, */
    /*                                                       const std::vector<larcv::Image2D>& adc_wholeview_v, */
    /*                                                       const int padding=50 ) const; */
    
    /// within the region defined by the input images, provide a masked adc image
    std::vector< larcv::Image2D > makeCosmicMaskedImage(  const std::vector<larcv::Image2D>& adc_view_v ) const;

    /// within the region defined by the input images, tag the pixels with the cluster IDs
    std::vector< larcv::Image2D > makeClusterTaggedImages(  const std::vector<larcv::Image2D>& adc_view_v ) const;
    
    /// get the io manager (providing mutable instance)
    larlite::storage_manager& io() { return (*_io); };

    // utility function: make imagemeta bounding box from a pixelmask object
    static larcv::ImageMeta metaFromPixelMask( const larlite::pixelmask& mask,
                                               unsigned int planeid=0,
                                               const int padding=50,
                                               const larcv::ImageMeta* bounding_meta=nullptr);

    // utility function: make an image2d from a pixelmask, given the output meta size
    static larcv::Image2D image2dFromPixelMask( const larlite::pixelmask& mask, const larcv::ImageMeta& outputmeta );

    /// get the number of clusters in the current entry
    int getNumClusters() const;

    /// get current entry number
    int getCurrentEntry() { return _entry; };

    std::vector< larcv::Image2D >& getWholeViewDLOutputImage( DLCosmicTagUtil::DLOutput_t dltype ); 
    
  protected:

    // configuration parameters
    std::vector<std::string> _larlite_input_filename_v;
    std::vector<std::string> _intime_pixelmask_producername_v;
    std::string _intime_larflowcluster_producername;
    std::string _ssnet_shower_producername;
    std::string _ssnet_track_producername;
    std::string _ssnet_endpt_producername;
    std::string _infill_producername;

    /// Class configured
    bool fConfigured;

    // larlite interface members
    size_t _entry; //< current larlite input entry
    larlite::storage_manager* _io; //< interface to larlite file
    const larlite::event_larflowcluster* m_larflowcluster_v; //< event container for larlite larflow clusters
    std::vector<const larlite::event_pixelmask*> m_pixelmask_vv; //< event containers for larlite pixelmask. one for each plane.
    std::vector<const larlite::event_pixelmask*> m_dloutput_masks_v; //< event containers for DL output pixelmasks

    // entry loaded
    bool fEntryLoaded;

    // whole image dloutput images from mask. cached to avoid remaking wholeimage too many times
    int _dloutput_entry;
    std::vector< std::vector<larcv::Image2D> > m_dloutput_wholeview_vv;

    static const larcv::logger& get_logger();

  private:

    // hack to allow us to user the LARCV LOGGER
    // we have a static instance to which we send output messages
    static DLCosmicTagUtil* _g_logger_instance;
    
  };

}

#endif
