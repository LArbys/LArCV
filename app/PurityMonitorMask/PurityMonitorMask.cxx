#include "PurityMonitorMask.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  PurityMonitorMask::PurityMonitorMask()
    : larcv::ProcessBase("PurityMonitorMask")
  {
  }

  PurityMonitorMask::~PurityMonitorMask() {}

  void PurityMonitorMask::configure( const PSet& pset ) {}

  void PurityMonitorMask::initialize() {
    ana_file().cd();

    // create anafile
    _tree = new TTree("pmnoise","Purity Monitor Noise variables");
    _tree->Branch("run",      &_run,      "run/I");
    _tree->Branch("subrun",   &_subrun,   "subrun/I");
    _tree->Branch("event",    &_event,    "event/I");
    _tree->Branch("row",      &_row,      "row/I");
    _tree->Branch("qsum",     &_qsum,     "qsum/F");
    _tree->Branch("naboveth", &_naboveth, "naboveth/I" );
    
  }

  bool PurityMonitorMask::process( IOManager& mgr ) {

    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,"wire");

    _run    = mgr.event_id().run();
    _subrun = mgr.event_id().subrun();
    _event  = mgr.event_id().event();

    std::vector<larcv::Image2D> masked_v;
    for ( auto const& img : ev_adc->Image2DArray() ) {
      larcv::Image2D newimg( img.meta() );
      newimg.paint(0.0);
      masked_v.emplace_back( std::move(newimg) );
    }

    process( ev_adc->Image2DArray(), masked_v, 10.0 );
    
    return true;
  }

  /**
   * process adc image, look for purity monitor noise signature
   *
   */
  bool PurityMonitorMask::process( const std::vector<larcv::Image2D>& adc_v,
                                   std::vector<larcv::Image2D>& masked_v,
                                   const float threshold ) {

    for ( size_t p=0; p<3; p++ ) {
      const larcv::ImageMeta& meta = adc_v[p].meta();
      const larcv::Image2D& img = adc_v[p];
      for (size_t r=0; r<meta.rows(); r++ ) {

        float chargesum     = 0.;
        int   nabove_thresh = 0;
        for ( size_t c=0; c<meta.cols();c++ ) {
          float pix = img.pixel(r,c);
          if ( pix>threshold ) {
            chargesum+=img.pixel(r,c);
            nabove_thresh += 1;
          }
        }
          
        // fill-ana for study
        _row = (int)r;
        _qsum = chargesum;
        _naboveth = nabove_thresh;
        _tree->Fill();
      }
    }

    
    return true;
  }

  void PurityMonitorMask::finalize() {
    
  }

}
