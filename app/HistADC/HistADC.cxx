#ifndef __HISTADC_CXX__
#define __HISTADC_CXX__

#include "HistADC.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/ROI.h"
#include "DataFormat/EventROI.h"

namespace larcv {

  static HistADCProcessFactory __global_HistADCProcessFactory__;

  HistADC::HistADC(const std::string name)
    : ProcessBase(name)
  {}
    
  void HistADC::configure(const PSet& cfg)
  {
    fHiResCropProducer = cfg.get<std::string>("HiResCropProducer");
    fROIProducer       = cfg.get<std::string>("ROIProducer");
    fPlane0Thresh      = cfg.get<int>("Plane0Thresh");
    fPlane1Thresh      = cfg.get<int>("Plane1Thresh");
    fPlane2Thresh      = cfg.get<int>("Plane2Thresh");
    fFillCosmic        = cfg.get<bool>("FillCosmic");
  }

  void HistADC::initialize()
  {
    // clear the vector
    for (auto &p : m_hADC_v ) {
      delete p;
      p = NULL;
    }
    m_hADC_v.clear();

    m_tree = new TTree("adctree","ADC/pixel summary of cropped regions");
    m_tree->Branch( "plane",&m_plane,"plane/I");
    m_tree->Branch( "isneutrino",&m_neutrino,"isneutrino/I");
    m_tree->Branch( "npixels",&m_npixels,"npixels/I");
    m_tree->Branch( "adcsum",&m_sum,"adcsum/F");
    m_tree->Branch( "edep",&m_edep,"edep/F");
    
  }

  bool HistADC::process(IOManager& mgr)
  {

    auto event_image = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,fHiResCropProducer));
    auto event_roi         = (larcv::EventROI*)(mgr.get_data(kProductROI,fROIProducer));

    auto roi = event_roi->at(0);

    // if ( fFillCosmic && roi.Type()!=kROICosmic) 
    //   return true;
    // if ( !fFillCosmic && roi.Type()==kROICosmic )
    //   return true;

    if ( roi.Type()!=kROICosmic ) {
      m_neutrino = 1;
      m_edep = roi.EnergyDeposit();
    }
    else {
      m_neutrino = 0;
      m_edep = 0.;
    }


    //m_neutrino = ( fFillCosmic ) ? 0 : 1;

    for ( auto const& img: event_image->Image2DArray() ) {
      auto const plane = img.meta().plane();
      if(m_hADC_v.size() <= plane) m_hADC_v.resize(plane+1,nullptr);
      

      // if(!m_hADC_v[plane]) {
      // 	if (fFillCosmic )
      // 	  m_hADC_v[plane] = new TH1D(Form("hADCsum_%s_Plane%02hu_cosmic",fHiResCropProducer.c_str(),plane),
      // 				     Form("ADC Values for Plane %hu",plane),
      // 				     500,0,10e3);
      // 	else
      // 	  m_hADC_v[plane] = new TH1D(Form("hADCsum_%s_Plane%02hu_neutrino",fHiResCropProducer.c_str(),plane),
      // 				     Form("ADC Values for Plane %hu",plane),
      // 				     500,0,10e3);
      // }
	
	
      m_sum = 0.0;
      m_npixels = 0;
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
	  m_sum += adc;
	  m_npixels++;
	}
      }

      //auto& p_hADC = m_hADC_v[plane];
      //std::cout << sum << std::endl;
      //p_hADC->Fill( m_sum );
      m_plane = plane;
      m_tree->Fill();
    }//end of loop over plane image
    return true;
  }

  void HistADC::finalize()
  {
    if ( has_ana_file() ) {
      //for(auto& p : m_hADC_v) if(p) p->Write();
      m_tree->Write();
    }
  }

}
#endif
