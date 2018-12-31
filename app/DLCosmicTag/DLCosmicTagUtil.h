#ifndef __DLCosmicTagUtil_h__
#define __DLCosmicTagUtil_h__

#include <vector>

// LArCV
#include "Base/larcv_base.h"
#include "Base/PSet.h"
#include "DataFormat/Image2D.h"


// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/pixelmask.h"
#include "DataFormat/larflowcluster.h"

namespace larcv {

  class DLCosmicTagUtil : public larcv::larcv_base {
  public:
    
    DLCosmicTagUtil();
    virtual ~DLCosmicTagUtil();

    void Configure( PSet& pset );
    //void go_to_event( int run, int subrun, int event ); // to-do
    void go_to_entry( size_t entry );

    larlite::event_base* get_data( larlite::data::DataType_t data_type, std::string producername );

    /// use the intime cluster objects to form a crop within the wholeview image
    std::vector< larcv::Image2D > makeIntimeCroppedImage( const std::vector<larcv::Image2D>& adc_wholeview_v, const int padding=10 ) const;
    
    /// within the region defined by the input images, provide a masked adc image
    std::vector< larcv::Image2D > makeCosmicMaskedImage(  const std::vector<larcv::Image2D>& adc_view_v ) const;

    /// within the region defined by the input images, tag the pixels with the cluster IDs
    std::vector< larcv::Image2D > makeClusterTaggedImages(  const std::vector<larcv::Image2D>& adc_view_v ) const;

    /// get the io manager (providing mutable instance)
    larlite::storage_manager& io() { return (*_io); };

    // utility function: make imagemeta bounding box from a pixelmask object
    static larcv::ImageMeta metaFromPixelMask( const larlite::pixelmask& mask, unsigned int planeid=0 );

    /// get the number of clusters in the current entry
    int getNumClusters() const;    
    
  protected:

    // configuration parameters
    std::string _larlite_input_filename;
    std::vector<std::string> _intime_pixelmask_producername_v;
    std::string _intime_larflowcluster_producername;

    // larlite interface members
    size_t _entry; //< current larlite input entry
    larlite::storage_manager* _io; //< interface to larlite file
    const larlite::event_larflowcluster* m_larflowcluster_v; //< event container for larlite larflow clusters
    std::vector<const larlite::event_pixelmask*> m_pixelmask_vv; //< event containers for larlite pixelmask. one for each plane.
    bool fEntryLoaded;
  };

}

#endif
