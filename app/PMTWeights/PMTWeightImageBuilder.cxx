#ifndef __PMTWEIGHTIMAGEBUILDER_CXX__
#define __PMTWEIGHTIMAGEBUILDER_CXX__

#include "PMTWeightImageBuilder.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {
  namespace pmtweights {

    static PMTWeightImageBuilderProcessFactory __global_PMTWeightImageBuilderProcessFactory__;
    

    PMTWeightImageBuilder::~PMTWeightImageBuilder() {
      delete m_WireWeights;
      m_WireWeights = NULL;
    }

    PMTWeightImageBuilder::PMTWeightImageBuilder(const std::string name)
      : ProcessBase(name)
    {}
    
    void PMTWeightImageBuilder::configure(const PSet& cfg)
    {
      fGeoFile     = cfg.get<std::string> ( "GeoFile" );
      fNWirePixels = cfg.get<int>( "NWirePixels" );
      fPMTImageProducer = cfg.get<std::string>( "PMTImageProducer" );
      fTPCImageProducer = cfg.get<std::string>( "TPCImageProducer" );
      fStartTick = cfg.get<int>( "StartTick" );
      fEndTick = cfg.get<int>( "EndTick" );
      fCheckSat = cfg.get<bool>("CheckSaturation");
      fPMTImageIndex = cfg.get<int>("PMTImageIndex");
    }
    
    void PMTWeightImageBuilder::initialize()
    {
      m_WireWeights = new PMTWireWeights( fGeoFile, fNWirePixels );
    }

    
    bool PMTWeightImageBuilder::process(IOManager& mgr)
    {

      auto pmt_event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,fPMTImageProducer));
      if(!pmt_event_image || pmt_event_image->Image2DArray().empty()) return false;      

      auto tpc_event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,fTPCImageProducer));
      if(!tpc_event_image || tpc_event_image->Image2DArray().empty()) return false;      


      // -------------------
      // Pick out PMT Image
      // -------------------
      larcv::Image2D const& pmtimg_highgain = pmt_event_image->at( fPMTImageIndex ); // placeholder code
      //larcv::Image2D const& pmtimg_lowgain  = event_image->at( 4 ); // placeholder code
      std::cout << "pmt image: " << pmtimg_highgain.meta().rows() << " x " << pmtimg_highgain.meta().cols() << std::endl;
      std::cout << "tpc image: " << tpc_event_image->at(2).meta().rows() << " x " << tpc_event_image->at(2).meta().cols() << std::endl;
      std::cout << "summing between: " << fStartTick << " and " << fEndTick << std::endl;
      // sum pmt charge in the trigger window

      larcv::Image2D pmtq( 32, 1 );
      float totalq = 0.0;
      for (int ipmt=0; ipmt<32; ipmt++) {
	float high_q = 0.0;
	float low_q  = 0.0;
	bool usehigh = true;
	for (int t=fStartTick; t<=fEndTick; t++) {
	  // sum over the trigger window
	  float hq = pmtimg_highgain.pixel( t, ipmt ); 
	  high_q += hq;
// 	  if ( fCheckSat ) {
// 	    low_q += pmt_event_image->at(fPMTImageIndex+1).pixel( t, ipmt );
// 	  }
	  if ( hq>1040 )
	    usehigh = false;
	}
	if ( high_q > 0.0 ) {
	  if ( usehigh ) {
	    pmtq.set_pixel( ipmt, 0, high_q/100.0 );
	    totalq += high_q/100.0;
	  }
	  else {
	    pmtq.set_pixel( ipmt, 0, 10.0*low_q/100.0 );
	    totalq += 10.0*low_q/100.0;
	  }
	}
      }//end of pmt loop
    
      // normalize charge
      if ( totalq > 0 ) {
	for (int ipmt=0; ipmt<32; ipmt++) {
	  pmtq.set_pixel( ipmt, 0, pmtq.pixel( ipmt, 1 )/totalq );
	}
      }
      std::cout << " make weight matrices" << std::endl;

      // make weight matrices
      std::vector<larcv::Image2D> pmtw_image_array;
      for (int p=0; p<3; p++) {
	larcv::Image2D const& planeweight = m_WireWeights->planeWeights[p];
	larcv::Image2D weights = planeweight*pmtq;
	pmtw_image_array.emplace_back( weights );
      }

      // now apply weights
      std::vector<larcv::Image2D> weighted_images;
      for (int p=0; p<3; p++) {
	larcv::Image2D const& tpcimg  = tpc_event_image->at( p );
	larcv::Image2D const& weights = pmtw_image_array.at( p );
	larcv::Image2D weighted( tpcimg );

	for (int c=0; c<weighted.meta().cols(); c++) {
	  float w = weights.pixel(c,0);
	  for (int r=0; r<weighted.meta().rows(); r++) {
	    weighted.set_pixel(r,c, w*tpcimg.pixel(r,c));
	  }
	}
	weighted_images.emplace_back( weighted );
      }
      
      auto out_image_v = (::larcv::EventImage2D*)(mgr.get_data(::larcv::kProductImage2D,"pmtweighted"));
      //mgr.set_id(tpc_event_image->run(),tpc_event_image->subrun(),tpc_event_image->event());
      //mgr.set_id(0,0,0);
      out_image_v->Emplace(std::move(weighted_images));

      return true;
    }

//     larcv::Image2D& PMTWeightImageBuilder::getWeightedImage( int plane ) {
//       return m_weighted_images.at(plane);
//     }
    
    void PMTWeightImageBuilder::finalize(TFile* ana_file)
    {}

  }
}
#endif
