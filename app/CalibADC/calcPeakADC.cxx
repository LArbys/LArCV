#ifndef __CALCPEAKADC_CXX__
#define __CALCPEAKADC_CXX__

#include "calcPeakADC.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static calcPeakADCProcessFactory __global_calcPeakADCProcessFactory__;

  calcPeakADC::calcPeakADC(const std::string name)
    : ProcessBase(name)
  {}
    
  void calcPeakADC::configure(const PSet& cfg)
  {
    fImageProducer = cfg.get<std::string>("ImageProducerName");
    fThreshold = cfg.get<float>("PeakThreshold");
    fDeadtime  = cfg.get<float>("Deadtime");
    fNewCols   = cfg.get<int>("NewCols",-1);
    fNewRows   = cfg.get<int>("NewRows",-1);
  }

  void calcPeakADC::initialize()
  {
    ttree = new TTree("adc","");
    ttree->Branch("planeid",&planeid,"planeid/I");
    ttree->Branch("wireid",&wireid,"wireid/I");
    ttree->Branch("peak",&peakmax, "peak/F" );
  }

  bool calcPeakADC::process(IOManager& mgr)
  {

    auto event_images = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, fImageProducer );
    for ( auto const& img_src : event_images->Image2DArray() ) {
    //for ( int p=0; p<=8; p+=4 ) { // hack for 12 channel data
      auto const& img_src = event_images->Image2DArray().at(p);
      larcv::Image2D img( img_src );
      if ( fNewCols>0 || fNewRows>0 )
	img.compress( fNewRows, fNewCols ); //504, 864
      int wfms = img.meta().cols();
      int ticks = img.meta().rows();
      //std::cout << "img (wmfs,ticks) = (" << wfms << " x " << ticks << ")" << std::endl;
      
      for (int w=0; w<wfms; w++) {
	bool inpeak = false;
	float pmax = -1;
	int peakcenter = -1;
	std::vector<int> peakcenters;

	for (int t=0; t<ticks; t++) {
	  float y = img.pixel( t, w );
	  if (y < fThreshold) // below thresh, skip
	    continue;

	  if (!inpeak) {
	    if (peakcenters.size()==0 || peakcenters.at(peakcenters.size()-1)+fDeadtime<t ) {
	      inpeak = true;
	      pmax = y;
	      peakcenter = t;
	    }
	  }
	  else {
	    if (y>pmax) {
	      pmax = y;
	      peakcenter = t;
	    }
	    else {
	      inpeak = false;
	      wireid = w;
	      planeid = img.meta().plane();
	      peakmax = pmax;
	      peakcenters.push_back( peakcenter );
	      ttree->Fill();
	    }
	  }//end of if in peak
	}//end of tick loop
      }//end of wire loop
    }//end of img loop
  }//end of process
  
  void calcPeakADC::finalize()
  {
    ttree->Write();
  }

}
#endif
