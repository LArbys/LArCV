#ifndef __HIRESIMAGEDIVIDER_CXX__
#define __HIRESIMAGEDIVIDER_CXX__

#include "HiResImageDivider.h"
#include "DataFormat/EventImage2D.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>

#include "opencv/cv.h"
#include "opencv2/opencv.hpp"

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
      fDumpImages  = cfg.get<bool>( "DumpImages" );
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
	std::cout << "division entry " << entry << ": ";
	std::cout << " p0: [" << plane0[0] << "," << plane0[1] << "]";
	std::cout << " p1: [" << plane1[0] << "," << plane1[1] << "]";
	std::cout << " p2: [" << plane2[0] << "," << plane2[1] << "]";
	std::cout << " t: ["  << tickbounds[0] << "," << tickbounds[1] << "]";
	std::cout << std::endl;
	
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

      std::cout << "Vertex in ROI: " << roi.X() << ", " << roi.Y() << ", " << roi.Z() << std::endl;

      // now we crop out certain pieces
      // The input images
      auto input_event_images = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,fInputImageProducer));
      auto output_event_images = (larcv::EventImage2D*)(mgr.get_data( kProductImage2D,fOutputImageProducer) );
      cropEventImages( *input_event_images, vertex_div, *output_event_images );
      if ( fDumpImages ) {
	cv::Mat outimg;
	for (int p=0; p<3; p++) {
	  larcv::Image2D const& cropped = output_event_images->at( p );
	  if ( p==0 )
	    outimg = cv::Mat::zeros( cropped.meta().rows(), cropped.meta().cols(), CV_8UC3 ); 
	  for (int r=0; r<cropped.meta().rows(); r++) {
	    for (int c=0; c<cropped.meta().cols(); c++) {
	      int val = std::min( 255, (int)cropped.pixel(r,c) );
	      val = std::max( 0, val );
	      outimg.at< cv::Vec3b >(r,c)[p] = (unsigned int)val;
	    }
	  }
	}
	char testname[200];
 	sprintf( testname, "test_tpcimage_%zu.png", input_event_images->event() );
 	cv::imwrite( testname, outimg );
      }

      // Output Segmentation
      if ( fCropSegmentation ) {
	// the semantic segmentation is only filled in the neighboor hood of the interaction
	// we overlay it into a full image (and then crop out the division)
	auto input_seg_images = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,fInputSegmentationProducer));
	larcv::EventImage2D full_seg_images;
	for ( unsigned int p=0; p<3; p++ ) {
	  larcv::Image2D const& img = input_event_images->at( p ); 
	  larcv::ImageMeta seg_image_meta( img.meta().width(), img.meta().height(),
					   img.meta().rows(), img.meta().cols(),
					   img.meta().min_x(), img.meta().max_y(),
					   img.meta().plane() );
	  larcv::Image2D seg_image( seg_image_meta );
	  seg_image.paint( 0.0 );
	  seg_image.overlay( input_seg_images->at(p) );
	  full_seg_images.Emplace( std::move(seg_image) );
	}
	auto output_seg_images = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,fOutputSegmentationProducer));
	cropEventImages( full_seg_images, vertex_div, *output_seg_images );

	if ( fDumpImages ) {
	  cv::Mat outimg;
	  for (int p=0; p<3; p++) {
	    larcv::Image2D const& cropped = output_seg_images->at( p );
	    if ( p==0 )
	      outimg = cv::Mat::zeros( cropped.meta().rows(), cropped.meta().cols(), CV_8UC3 ); 
	    for (int r=0; r<cropped.meta().rows(); r++) {
	      for (int c=0; c<cropped.meta().cols(); c++) {
		int val = std::min( 255, (int)(cropped.pixel(r,c)+0.4)*10 );
		val = std::max( 0, val );
		outimg.at< cv::Vec3b >(r,c)[p] = (unsigned int)val;
	      }
	    }
	  }
	  char testname[200];
	  sprintf( testname, "test_seg_%zu.png", input_event_images->event() );
	  cv::imwrite( testname, outimg );
	}//if draw
      }// if crop seg
      
      // Output PMT weighted
//       if ( fCropPMTWeighted ) 
// 	cropEventImages( mgr, vertex_div, fInputPMTWeightedProducer, fOutputPMTWeightedProducer );
      
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
    
    void HiResImageDivider::cropEventImages( const larcv::EventImage2D& event_images, const larcv::hires::DivisionDef& div, larcv::EventImage2D& output_images ) { 

      // Output Image Container
      std::vector<larcv::Image2D> cropped_images;

      for ( auto const& img : event_images.Image2DArray() ) {
	int iplane = (int)img.meta().plane();
	larcv::ImageMeta const& divPlaneMeta = div.getPlaneMeta( iplane );
	// we adjust the actual crop meta
	int tstart = divPlaneMeta.max_y()-divPlaneMeta.height();
	int twidth = fMaxWireImageWidth*fTickDownSample;
	int tmax = std::min( tstart+twidth, (int)img.meta().max_y() );
	larcv::ImageMeta cropmeta( divPlaneMeta.width(), twidth,
				   divPlaneMeta.width(), twidth,
				   divPlaneMeta.min_x(), tmax );

	std::cout << "image: " << img.meta().height() << " x " << img.meta().width();
	std::cout << " t=[" << img.meta().min_y() << "," << img.meta().max_y() << "]"
		  << " wmin=" << img.meta().min_x();
	std::cout << std::endl;
	
	std::cout << "div: " << divPlaneMeta.height() << " x " << divPlaneMeta.width();
	std::cout << " t=[" << divPlaneMeta.min_y() << "," << divPlaneMeta.max_y() << "]"
		  << " wmin=" << divPlaneMeta.min_x();
	std::cout << std::endl;

	std::cout << "crop: " << cropmeta.height() << " x " << cropmeta.width();
	std::cout << " t=[" << cropmeta.min_y()  << "," << cropmeta.max_y() << "]"
		  << " wmin=" << cropmeta.min_x();
	
	std::cout << std::endl;

	Image2D cropped = img.crop( cropmeta );
	std::cout << "cropped." << std::endl;
	cropped.resize( fMaxWireImageWidth*fTickDownSample, fMaxWireImageWidth, 0.0 );  // resize to final image size (and zero pad extra space)
	std::cout << "resized." << std::endl;

	cropped.compress( (int)cropped.meta().height()/6, fMaxWireImageWidth, larcv::Image2D::kSum );
	std::cout << "downsampled. " << cropped.meta().height() << " x " << cropped.meta().width() << std::endl;
	
	
	cropped_images.emplace_back( cropped );
	std::cout << "stored." << std::endl;
      }//end of plane loop

      output_images.Emplace( std::move( cropped_images ) );

    }



  }
}
#endif
