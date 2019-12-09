#include "PurityMonitorMask.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPixel2D.h"

namespace larcv {

  static PurityMonitorMaskProcessFactory __global_PurityMonitorMaskProcessFactory__;
  
  PurityMonitorMask::PurityMonitorMask( std::string instance_name )
    : larcv::ProcessBase("PurityMonitorMask"),
    _name(instance_name)
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
      larcv::Image2D newimg( img );
      masked_v.emplace_back( std::move(newimg) );
    }

    std::vector<larcv::Pixel2DCluster> pixel_v(ev_adc->Image2DArray().size());
    
    process( ev_adc->Image2DArray(), masked_v, pixel_v, 10.0 );

    larcv::EventPixel2D* ev_pixout = (larcv::EventPixel2D*)mgr.get_data(larcv::kProductPixel2D, "pmmask" );
    for ( size_t p=0; p<ev_adc->Image2DArray().size(); p++ ) {
      ev_pixout->Emplace( ev_adc->Image2DArray()[p].meta().plane(), std::move(pixel_v[p]), ev_adc->Image2DArray()[p].meta() );
    }
    
    larcv::EventImage2D* ev_out = (larcv::EventImage2D*)mgr.get_data(larcv::kProductImage2D,"pmmask");
    ev_out->Emplace( std::move(masked_v) );

    
    
    return true;
  }

  /**
   * process adc image, look for purity monitor noise signature
   *
   */
  bool PurityMonitorMask::process( const std::vector<larcv::Image2D>& adc_v,
                                   std::vector<larcv::Image2D>& masked_v,
                                   std::vector<larcv::Pixel2DCluster>& pixel_v,
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

        if ( _naboveth>500 || (_naboveth>0 && _qsum/float(_naboveth)>400 ) ) {
          for ( size_t c=0; c<meta.cols();c++ ) {
            if ( c==0 ) {
              larcv::Pixel2D pix(c,r);
              pix.Intensity( masked_v[p].pixel(r,c) );
              pixel_v[p] += pix;
            }
            masked_v[p].set_pixel(r,c,0.0);
          }
        }
      }
    }
    
    
    return true;
  }

  void PurityMonitorMask::finalize() {
    ana_file().cd();
    _tree->Write();
  }

}
