#ifndef __HISTADC_CXX__
#define __HISTADC_CXX__

#include "HistADC.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static HistADCProcessFactory __global_HistADCProcessFactory__;

  HistADC::HistADC(const std::string name)
    : ProcessBase(name)
  {}
    
  void HistADC::configure(const PSet& cfg)
  {
    fHiResCropProducer = cfg.get<std::string>("HiResCropProducer");
    fPlane0Thresh      = cfg.get<int>("Plane0Thresh");
    fPlane1Thresh      = cfg.get<int>("Plane1Thresh");
    fPlane2Thresh      = cfg.get<int>("Plane2Thresh");
  }

  void HistADC::initialize()
  {
    // clear the vector
    for (auto &p : m_hADC_v ) {
      delete p;
      p = NULL;
    }
    m_hADC_v.clear();
    
  }

  bool HistADC::process(IOManager& mgr)
  {

    auto event_image = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,fHiResCropProducer));

    for ( auto const& img: event_image->Image2DArray() ) {
      auto const plane = img.meta().plane();
      if(m_hADC_v.size() <= plane) m_hADC_v.resize(plane+1,nullptr);
      
      if(!m_hADC_v[plane]) m_hADC_v[plane] = new TH1D(Form("hADCsum_%s_Plane%02hu",fHiResCropProducer.c_str(),plane),
						      Form("ADC Values for Plane %hu",plane),
						      500,0,10e3);

      float sum = 0.0;
      auto const& adc_v = img.as_vector();

      float thresh = 0;
      switch ( plane ) {
      case 0:
	thresh = fPlane0Thresh;
	break;
      case 1:
	thresh = fPlane1Thresh;
	break;
      case 2:
	thresh = fPlane2Thresh;
	break;
      }

      for ( auto const& adc : adc_v ) {
	if ( adc>thresh ) {
	  sum += adc;
	}
      }

      auto& p_hADC = m_hADC_v[plane];
      p_hADC->Fill( sum );
    }//end of loop over plane image
    return true;
  }

  void HistADC::finalize(TFile* ana_file)
  {
    if ( ana_file ) {
      ana_file->cd();
      for(auto& p : m_hADC_v) if(p) p->Write();
    }
  }

}
#endif
