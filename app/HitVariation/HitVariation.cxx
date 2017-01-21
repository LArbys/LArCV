#include "HitVariation.h"

#include <iostream>
#include <string>
#include <cmath>
#include <utility>
#include <ctime>
#include <random>


// config/storage: from LArCV
#include "Base/PSet.h"
#include "Base/LArCVBaseUtilFunc.h"

// larcv data
#include "DataFormat/DataFormatTypes.h"
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPixel2D.h"
#include "DataFormat/EventROI.h"

// ROOT
#include "TRandom3.h" //< ROOT is already a dependency so going to use it's RNG

namespace larcv {

  HitVariation::HitVariation( std::string configuration_filename ) {

    // we are going to do everything by configuration file. checkout varyhits.cfg for example

    // first we parse config file into PSet
    larcv::PSet toplevel_pset = larcv::CreatePSetFromFile( configuration_filename );
    larcv::PSet hitvary_pset  = toplevel_pset.get<larcv::PSet>("HitVariationConfig");

    // we get the IOManager configuration from a subblock of parameters
    larcv::PSet iomanconfig   = hitvary_pset.get<larcv::PSet>("IOManager");
    m_ioman = new IOManager( iomanconfig );
    m_ioman->initialize();

    m_nentries = m_ioman->get_n_entries();

    // now we set the algorithm's parameters
    m_config.overall_std     = hitvary_pset.get<double>("OverallStdDev");
    m_config.pixel_std       = hitvary_pset.get<double>("PixelStdDev");
    m_config.translation_std = hitvary_pset.get<double>("TranslationStdDev");
    m_config.amplitude_std   = hitvary_pset.get<double>("AmplitudeStdDev");
    m_config.width_std       = hitvary_pset.get<double>("WidthStdDev");
    m_config.pixel_tree_name = hitvary_pset.get<std::string>("PixelTreeName");
    m_config.originalimg_tree_name = hitvary_pset.get<std::string>("OriginalImageTreeName");
    m_config.roi_label_tree_name = hitvary_pset.get<std::string>("ROILabelTreeName");
    m_config.numplanes       = hitvary_pset.get<int>("NumberOfPlanes",3);
    m_config.make_random_image = hitvary_pset.get<bool>("MakeExampleRandomImages");

    // rng 
    m_seed = hitvary_pset.get<long>("RNGSeed",-1);

    if ( m_seed<0 ) {
      m_rng = new TRandom3( time(NULL) );
    }
    else
      m_rng = new TRandom3( m_seed );


  }

  HitVariation::~HitVariation() {
    // destructor
    delete m_ioman;
    delete m_rng;
  }


  void HitVariation::GenerateImages( const int num_images, const std::vector<double>& parameters, 
				     std::vector< std::vector<larcv::Image2D> >& output_orig,
				     std::vector< std::vector<larcv::Image2D> >& output_manip,
				     std::vector<int>& labels, std::vector<int>& entrynumbers ) {
    // Generates images from hits with parameterized variations
    // inputs
    // -------
    // num_images: number of images to generate
    // parameters: a vector of parameter values
    //
    // output
    // ------
    // vector< vector<imgs> >:  outer vector is for each event.  inner vector contains images for planes 0, 1, 2, etc....
    // vector<int>: 
    // 
    // this function can be called in python (w/ larcv and ROOT bindings)
    //  example: 
    //  import ROOT
    //  from ROOT import std
    //  from larcv import larcv
    //
    //  hitvaryalgo = larcv.HitVariation( configfile_path )
    //  labels = std.vector("int")() # this is how one makes stdlib vectors in PyROOT
    //  
    //  for batches
    //    event_img_v = hitvaryalgo.GenerateImages( batchsize, pars, labels, )
    //    for ibatch in batchsize:
    //      img_v = event_img_v.at(ibatch)
    //      for p in nplanes:
    //        img = larcv.as_ndarray( img_v.at(p)) )
    //   ...
    //   // do stuff


    output_orig.clear();
    output_manip.clear();
    
    // get random set of entries
    entrynumbers.resize( num_images );
    for (int ientry=0; ientry<num_images; ientry++){
      entrynumbers[ientry] = m_rng->Integer(m_nentries);
    }
    labels.resize(num_images);

    // get the data
    int ientry = 0;
    for ( auto &entry : entrynumbers ) {
      // set the entry in the file
      std::cout << "Reading entry " << entry << std::endl;
      m_ioman->read_entry( entry );

      // get image container
      larcv::EventImage2D* event_original_images = (larcv::EventImage2D*)m_ioman->get_data( larcv::kProductImage2D, m_config.originalimg_tree_name );
      const std::vector< larcv::Image2D >& input_images = event_original_images->Image2DArray();
      
      // output image container
      std::vector< larcv::Image2D > output_orig_event;
      std::vector< larcv::Image2D > output_manip_event;
      
      if ( !m_config.make_random_image ) {
	
	// get the pixel data
	std::cout << "Reading pixel containers" << std::endl;
	larcv::EventPixel2D* event_pixel_container = (larcv::EventPixel2D*)m_ioman->get_data( larcv::kProductPixel2D, m_config.pixel_tree_name );

	std::cout << "Making output images" << std::endl;
	//	for ( int p=0; p<m_config.numplanes; p++ ) {
	// const larcv::ImageMeta& meta = event_original_images->Image2DArray().at(p).meta(); //< meta describes image size and coordinates
	// const std::vector<larcv::Pixel2D>& pixels_v = event_pixel_container->Pixel2DArray( p ); //< vector containing pixels
	//larcv::Image2D plane_image = MakeImageFromHits( pixels_v, meta, parameters ); ///< proper function
	//output_images.emplace_back( std::move(plane_image) );
	//}
      }
      else {
	for (int p=0; p<(int)input_images.size(); p++ ) {
	  const larcv::ImageMeta& meta =input_images.at(p).meta(); //< meta describes image size and coordinates
	  std::vector<larcv::Pixel2D> dummy_pixel_v;
	  std:: vector<larcv::Image2D> plane_image = MakeRandomImage( dummy_pixel_v, meta, parameters );
	  output_orig_event.emplace_back( std::move(plane_image.at(0) ));
	  output_manip_event.emplace_back( std::move(plane_image.at(1) ));
				    
	}
      }

      // save it!
      output_orig.emplace_back( std::move(output_orig_event) );
      output_manip.emplace_back( std::move(output_manip_event) );
      
      // Get the truth label. Stored in the ROI objects which store event/interaction/particle meta data
      std::cout << "get truth info to make label" << std::endl;
      larcv::EventROI* event_roi_metadata = (larcv::EventROI*)m_ioman->get_data( larcv::kProductROI, m_config.roi_label_tree_name );
      const larcv::ROI& roi = event_roi_metadata->ROIArray().front();
      int label = -1;

      // The label enums are in core/DataFormats/DataFormatTypes.h
      switch ( roi.Type() ) {
        case larcv::kROICosmic:
          label = 0;
          break;
        case larcv::kROIBNB:
          label = 1;
          break;
        default:
          throw std::runtime_error("Unrecognized ROI Type.");
          break;
      }

      labels[ientry] = label;
      ientry++;
    }//end of event loop
  }//end of GenerateImages

  larcv::Image2D HitVariation::MakeImageFromHits( const std::vector<larcv::Pixel2D>& pixels, const larcv::ImageMeta& meta, const std::vector<double>& parameters ) {
    // Take a vector of this and turn into an image

    larcv::Image2D output_image( meta ); ///< make an image with the same size and coorindates as the original image
    output_image.paint(0.0); ///< set all pixels to zero

    // feel free to use STL RNG's here
    // std::normal_distribution<double> overal_noise(1.0,overall_std);
    // std::normal_distribution<double> pixel_noise(1.0, pixel_std);
    // std::uniform_real_distribution<double> picker(0.0,1.0);

    // double o_noise = overall_noise(generator);
    // double p_noise = pixel_noise(generator);
    // double selecter1 = picker(generator);
    // double selecter2 = picker(generator) ;

    // alternatively, m_rng class member can give you
    // m_rng->Gaus( mean, sigma )
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> overall_noise(1.0,m_config.overall_std);
    std::uniform_real_distribution<double> picker(0.0,1.0);

    for ( auto const& pixel : pixels ) {
      int row = pixel.X();
      int col = pixel.Y();
      float val = pixel.Intensity();
      float manip = overall_noise(gen);
      float choice1 =  picker(gen);
      float choice2 =  picker(gen);
      val = val*manip*100;
      if (choice1>0.98){row = row+1;}
      if (choice1>0.995){row = row+1;}
      if (choice2<0.02){col = col+1;}
      if (choice2>0.005){col = col+1;}

      if (true){ //(2<=pixel.Width()){
	
      for (int i=0; i<pixel.Width(); i++)
      {
	row = row+i;
      if (row>=meta.rows()){row=meta.rows()-1;}
      if (row<0){row=0;}
      if (col>=meta.cols()){col=meta.cols()-1;}
      if (col<0){col=0;}

      output_image.set_pixel(row, col, val);

      }
      }
      // you won't need this right away I think, but this is how you convert to time and wire
      //float tick = meta.pos_y( row );
      //float wire = meta.pos_x( col );

      // MATT's FUNCTIONS HERE
    }

    return output_image;

    // LEFT OVER STUFF.
 //    int nentries = 10;

 //    for (int ientry=0; ientry<nentries; ientry++) {

 //    for ( auto &img : event_imgs->Image2DArray() ) {   // what exactly is this looping over??
 //      const larcv::ImageMeta& meta = img.meta(); // get a constant reference to the Image's meta data (which defines the image size)

 //      // get actual original image because I need to output it

 //      //somehow get all my pixel2d objects

 //      // this make the new image that I'm gonna fill
 //      larcv::Image2D newimg( meta ); // make a new image using the old meta. this means, the size and coordinate system is the same.



 //      // so I want to randomly decide if I want from signal or noise and give it a 1 and 0 label

 //      // also, how do I make these come out as a 4 dimensional tensor like [batch_size, col, row, rgb]



 //      // I probably wanna loop through the pixel objects right? something like this

 //      //do I want to considering dropping the first or last pixel? maybe give a drop probability thats like 0.01? 

 //      // for hit in pixel2d
 //      // row  = row value
 //      // cen = center
 //      o_noise = overall_noise(generator);
 //      // overall = get overall intensity noise        this has its own parameter
 //      // for i in length of hit
 //      p_noise = pixel_noise(generator);
 //      selecter1 = picker(generator);
 //      selecter2 = picker(generator);
 //      // get local pixel noise                        this has its own parameter
 //      // decide if I want to move it and how much     this has its own parameter
 //      // set pixel (colval, maybe changed row val, pixel value)
      
      

 //      // so i do this for all of them
 //      // use Kazus function to turn image2d to numpy array
 //      // return orig and manip w label

 //      for (int irow=0; irow<meta.rows(); irow++) {
	// for (int icol=0; icol<meta.cols(); icol++) {
	//   float newval = img.pixel(irow,icol)*rand.Uniform(); // generate a random value based on the old image value
	//   newimg.set_pixel( irow, icol, newval ); // set the value int he new image
	//   if ( img.pixel(irow,icol)>0 && newval/img.pixel(irow,icol)>0.5 ) {
	//     // if it's above 0.5, we make a Pixel2D object
	//     larcv::Pixel2D hit( icol, irow );
	//     hit.Intensity( newval );
	//     hit.Width( 1.0 );
	//     // store the pixel 2D in its output container
	//     event_pixel2d->Emplace( (larcv::PlaneID_t)meta.plane(), std::move(hit) );
	//   }
	// }
 //      }
 //      // put the new image into the image output container
 //      output_container.emplace_back( std::move(newimg) ); // we use emplace and move so that we avoid making a copy
 //    }
 //    }

  }//end of MakeImageFromHits

  std::vector<larcv::Image2D> HitVariation::MakeRandomImage( const std::vector<larcv::Pixel2D>& pixels, const larcv::ImageMeta& meta, const std::vector<double>& parameters ) {
    // create new image from meta. the latter provides the dimensions and coordinates of image
    std::vector<larcv::Image2D> output;
    // this is manip  
    larcv::Image2D img(meta);
    // set all pixels to zero
    img.paint(0.0);
    for ( size_t row=0; row<meta.rows(); row++) {
      for ( size_t col=0; col<meta.cols(); col++) {
	float random_value = m_rng->Gaus( 50.0, 20.0 );
	img.set_pixel( row, col, random_value );
      }
    }
    larcv::Image2D img_orig(meta);
    // set all pixels to zero
    img_orig.paint(0.0);
    for ( size_t row=0; row<meta.rows(); row++) {
      for ( size_t col=0; col<meta.cols(); col++) {
	float random_value = m_rng->Gaus( 50.0, 20.0 );
	img_orig.set_pixel( row, col, random_value );
      }
    }

    output.emplace_back(std::move(img_orig));
    output.emplace_back(std::move(img));
    
    
    return output;
  }

}//end of larcv namespace

