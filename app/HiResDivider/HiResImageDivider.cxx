#ifndef __HIRESIMAGEDIVIDER_CXX__
#define __HIRESIMAGEDIVIDER_CXX__

#include "HiResImageDivider.h"
#include "DataFormat/EventImage2D.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>

namespace larcv {
  namespace hires {
    static HiResImageDividerProcessFactory __global_HiResImageDividerProcessFactory__;
    
    HiResImageDivider::HiResImageDivider(const std::string name)
      : ProcessBase(name)
    {}
    
    void HiResImageDivider::configure(const PSet& cfg)
    {
      fDivisionFile       = cfg.get<std::string>("DivisionFile");
      fNPlanes            = cfg.get<int>( "NPlanes", 3 );
      fTickStart          = cfg.get<int>( "TickStart", 2400 );
      fTickDownSample     = cfg.get<int>( "TickDownSampleFactor", 6 );
      fMaxWireImageWidth  = cfg.get<int>( "MaxWireImageWidth" );
      fInputROIProducer   = cfg.get<std::string>( "InputROIProducer" );
      fNumNonVertexDivisionsPerEvent = cfg.get<int>( "NumNonVertexDivisionsPerEvent" );
      fInputImageProducer = cfg.get<std::string>( "InputImageProducer" );
      fOutputImageProducer = cfg.get<std::string>( "OutputImageProducer" );
      fInputSegmentationProducer = cfg.get<std::string>( "InputSegmentationProducer" );
      fOutputSegmentationProducer = cfg.get<std::string>( "OutputSegmentationProducer" );
      fInputPMTWeightedProducer = cfg.get<std::string>( "InputPMTWeightedProducer" );
      fOutputPMTWeightedProducer = cfg.get<std::string>( "OutputPMTWeightedProducer" );
      fCropSegmentation = cfg.get<bool>( "CropSegmentation" );
      fCropPMTWeighted  = cfg.get<bool>( "CropPMTWeighted" );

    }
    
    void HiResImageDivider::initialize()
    {
      // The image divisions are calculated before hand in the fixed grid model
      // we load the prefined region image definitions here
      
      TFile* f = new TFile( fDivisionFile.c_str(), "open" );
      TTree* t = (TTree*)f->Get("imagedivider/regionInfo");
      int **planebounds = new int*[fNPlanes];
      int planenwires[fNPlanes];
      for (int p=0; p<fNPlanes; p++) {
	planebounds[p] = new int[2];
	char bname1[100];
	sprintf( bname1, "plane%d_wirebounds", p );
	t->SetBranchAddress( bname1, planebounds[p] );

	char bname2[100];
	sprintf( bname2, "plane%d_nwires", p );
	t->SetBranchAddress( bname2, &(planenwires[p]) );
	//std::cout << "setup plane=" << p << " branches" << std::endl;
      }
      
      float zbounds[2];
      float xbounds[2];
      float ybounds[2];
      int tickbounds[2];

      t->SetBranchAddress( "zbounds", zbounds );
      t->SetBranchAddress( "ybounds", ybounds );
      t->SetBranchAddress( "xbounds", xbounds );
      t->SetBranchAddress( "tickbounds", tickbounds );

      fMaxWireInRegion = 0;
      size_t entry = 0;
      size_t bytes = t->GetEntry(entry);
      while ( bytes>0 ) {
	for (int p=0; p<3; p++) {
	  if ( fMaxWireInRegion<planenwires[p] )
	    fMaxWireInRegion = planenwires[p];
	}
	int plane0[2], plane1[2], plane2[2];
	for (int i=0; i<2; i++) {
	  plane0[i] = (int)planebounds[0][i];
	  plane1[i] = (int)planebounds[1][i];
	  plane2[i] = (int)planebounds[2][i];
	  tickbounds[i] *= fTickDownSample;
	  tickbounds[i] += fTickStart;
	}
	
	DivisionDef div( plane0, plane1, plane2, tickbounds, xbounds, ybounds, zbounds );
	
	m_divisions.emplace_back( div );
	entry++;
	bytes = t->GetEntry(entry);
	//std::cout << "Division tree entry:" << entry << " (" << bytes << ")" << std::endl;
      }

      if ( fMaxWireInRegion>fMaxWireImageWidth )
	fMaxWireImageWidth = fMaxWireInRegion;

      for (int p=0; p<fNPlanes; p++) {
	delete [] planebounds[p];
      }
      delete [] planebounds;
      
      f->Close();
      
    }
    
    bool HiResImageDivider::process(IOManager& mgr)
    {
      // This processor does the following:
      // 1) read in hi-res images (from producer specified in config)
      // 2) (how to choose which one we clip?)

      // we get the ROI which will guide us on how to use the image
      auto event_roi = (larcv::EventROI*)(mgr.get_data(larcv::kProductROI,fInputROIProducer));

      larcv::ROI roi;
      for ( auto const& aroi : event_roi->ROIArray() ) {
	if ( aroi.Type()==kROIBNB ) {
	  roi = aroi;
	}
      }
      
      if ( !isInteresting( roi ) )
	return false;

      // first we find the division with a neutrino in it
      int idiv = findVertexDivision( roi );
      if ( idiv==-1 ) {
	LARCV_ERROR() << "No divisions were found that contained an event vertex." <<std::endl;
      }
      larcv::hires::DivisionDef const& vertex_div = m_divisions.at( idiv );

      // now we crop out certain pieces
      // The input images
      cropEventImages( mgr, vertex_div, fInputImageProducer, fOutputImageProducer );

      // Output Segmentation
      if ( fCropSegmentation )
	cropEventImages( mgr, vertex_div, fInputSegmentationProducer, fOutputSegmentationProducer );

      // Output PMT weighted
      if ( fCropPMTWeighted ) 
	cropEventImages( mgr, vertex_div, fInputPMTWeightedProducer, fOutputPMTWeightedProducer );
      
      return true;
    }
    
    void HiResImageDivider::finalize(TFile* ana_file)
    {}

    // -------------------------------------------------------

    bool HiResImageDivider::isInteresting( const larcv::ROI& roi ) {
      return true;
    }

    int HiResImageDivider::findVertexDivision( const larcv::ROI& roi ) {
      int regionindex = 0;
      for ( std::vector< larcv::hires::DivisionDef >::iterator it=m_divisions.begin(); it!=m_divisions.end(); it++) {
	DivisionDef const& div = (*it);
	if ( div.isInsideDetRegion( roi.X(), roi.Y(), roi.Z() ) )
	  return regionindex;
	regionindex++;
      }
      return -1;
    }
    
    bool HiResImageDivider::keepNonVertexDivision( const larcv::ROI& roi ) {
      return true;
    }

    void HiResImageDivider::cropEventImages( IOManager& mgr, const larcv::hires::DivisionDef& div, std::string producername, std::string outproducername ) {
      auto event_images = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,producername));
      // Output Image Container
      std::vector<larcv::Image2D> cropped_images;

      for ( auto const& img : event_images->Image2DArray() ) {
	int iplane = (int)img.meta().plane();
	larcv::ImageMeta const& divPlaneMeta = div.getPlaneMeta( iplane );
	// we adjust the actual crop meta
	larcv::ImageMeta cropmeta( divPlaneMeta.width(), fMaxWireImageWidth*fTickDownSample,
				   divPlaneMeta.width(), fMaxWireImageWidth*fTickDownSample,
				   divPlaneMeta.min_x(), divPlaneMeta.min_y() );
	Image2D cropped = img.crop( cropmeta );
	cropped.resize( fMaxWireImageWidth, cropped.meta().width(), 0.0 );  // resize to final image size (and zero pad extra space)
	cropped_images.emplace_back( cropped );
      }

      // insert
      auto output_images = (larcv::EventImage2D*)(mgr.get_data( kProductImage2D, outproducername) );
      output_images->Emplace( std::move( cropped_images ) );
    }



  }
}
#endif
