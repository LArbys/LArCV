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
      fImageProducer = cfg.get<std::string>( "ImageProducer" );
      fStartTick = cfg.get<int>( "StartTick" );
      fEndTick = cfg.get<int>( "StartTick" );
      fCheckSat = cfg.get<bool>("CheckSaturation");

    }
    
    void PMTWeightImageBuilder::initialize()
    {
      m_WireWeights = new PMTWireWeights( fGeoFile, fNWirePixels );
    }

    
    bool PMTWeightImageBuilder::process(IOManager& mgr)
    {

      m_pmtw_image_array.clear();

      auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,fImageProducer));
      if(!event_image || event_image->Image2DArray().empty()) return false;      


      // -------------------
      // Pick out PMT Image
      // -------------------
      larcv::Image2D const& pmtimg_highgain = event_image->at( 3 ); // placeholder code
      //larcv::Image2D const& pmtimg_lowgain  = event_image->at( 4 ); // placeholder code

      // sum pmt charge in the trigger window

      larcv::Image2D pmtq( 32, 1 );
      float totalq = 0.0;
      for (int ipmt=0; ipmt<32; ipmt++) {
	float high_q = 0.0;
	float low_q  = 0.0;
	bool usehigh = true;
	for (int t=fStartTick; t<=fEndTick; t++) {
	  // sum over the trigger window
	  float hq = pmtimg_highgain.pixel( t, 10*ipmt ); 
	  high_q += hq;
	  if ( fCheckSat ) {
	    low_q += event_image->at(4).pixel( t, ipmt );
	  }
	  if ( hq>1040 )
	    usehigh = false;
	}
	if ( high_q > 0.0 ) {
	  if ( usehigh ) {
	    pmtq.set_pixel( ipmt, 1, high_q/100.0 );
	    totalq += high_q/100.0;
	  }
	  else {
	    pmtq.set_pixel( ipmt, 1, 10.0*low_q/100.0 );
	    totalq += 10.0*low_q/100.0;
	  }
	}
      }//end of pmt loop
    
      // normalize charge
      if ( totalq > 0 ) {
	for (int ipmt=0; ipmt<32; ipmt++) {
	  pmtq.set_pixel( ipmt, 1, pmtq.pixel( ipmt, 1 )/totalq );
	}
      }
    
      for (int p=0; p<3; p++) {
	larcv::Image2D const& planeweight = m_WireWeights->planeWeights[p];
	larcv::Image2D weights = planeweight*pmtq;
	m_pmtw_image_array.emplace_back( weights );
      }
      
      
//       std::cout << "ww: " << wireweights[2] << std::endl;

      
    }
    
    void PMTWeightImageBuilder::finalize(TFile* ana_file)
    {}

  }
}
#endif
