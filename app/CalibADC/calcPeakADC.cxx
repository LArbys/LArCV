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
  {}

  void calcPeakADC::initialize()
  {
    out = new TFile("output_calcpeakadc.root","recreate");
    ttree = new TTree("adc","");
    ttree->Branch("planeid",&planeid,"planeid/I");
    ttree->Branch("wireid",&wireid,"wireid/I");
    ttree->Branch("peak",&peakmax, "peak/F" );
  }

  bool calcPeakADC::process(IOManager& mgr)
  {
    auto event_images = (larcv::EventImage2D*)mgr.get_data( larcv::kProductImage2D, "tpc" );
    for ( auto const& img : event_images->Image2DArray() ) {
      int wfms = img.meta().rows();
      int ticks = img.meta().cols();
      
      for (int w=0; w<wfms; w++) {
	bool inpeak = false;
	int pmax = -1;
	int peakcenter = -1;
	std::vector<int> peakcenters;

	for (int t=0; t<ticks; t++) {
	  float y = img.pixel( w, t );
	  if (y < 40) // below thresh, skip
	    continue;

	  if (!inpeak) {
	    if (peakcenters.size()==0 || peakcenters.at(-1)+20<t ) {
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
