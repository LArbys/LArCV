#include "HitVariation.h"

#include <iostream>
#include <cmath>
#include <utility>
#include <ctime>
#include <random>


// config/storage: from LArCV
#include "Base/PSet.h"
#include "Base/LArCVBaseUtilFunc.h"
#include "Base/DataCoordinator.h"

// larcv data
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPixel2D.h"


/// how to import kazus "to numpy" function


// how to have the function take inputs in python



namespace larcv {

  HitVariation::HitVariation() {

    // larlite
    dataco.add_inputfile( "larlite_opreco_0000.root", "larlite" );
    // larcv
    dataco.add_inputfile( "supera_data_0000.root", "larcv" );

    // configure
    dataco.configure( "config.cfg", "StorageManager", "IOManager", "ExampleConfigurationFile" );
  
    // initialize
    dataco.initialize();
    
    std::default_random_engine generator;
    std::normal_distribution<double> overal_noise(1.0,overall_std);
    std::normal_distribution<double> pixel_noise(1.0, pixel_std);
    std::uniform_real_distribution<double> picker(0.0,1.0);

    double o_noise = overall_noise(generator);
    double p_noise = pixel_noise(generator);
    double selecter1 = picker(generator);
    double selecter2 = picker(generator);



    int nentries = 10;

    for (int ientry=0; ientry<nentries; ientry++) {

    for ( auto &img : event_imgs->Image2DArray() ) {   // what exactly is this looping over??
      const larcv::ImageMeta& meta = img.meta(); // get a constant reference to the Image's meta data (which defines the image size)

      // get actual original image because I need to output it

      //somehow get all my pixel2d objects

      // this make the new image that I'm gonna fill
      larcv::Image2D newimg( meta ); // make a new image using the old meta. this means, the size and coordinate system is the same.



      // so I want to randomly decide if I want from signal or noise and give it a 1 and 0 label

      // also, how do I make these come out as a 4 dimensional tensor like [batch_size, col, row, rgb]



      // I probably wanna loop through the pixel objects right? something like this
      




      //do I want to considering dropping the first or last pixel? maybe give a drop probability thats like 0.01? 

      // for hit in pixel2d
      // row  = row value
      // cen = center
      o_noise = overall_noise(generator);
      // overall = get overall intensity noise        this has its own parameter
      // for i in length of hit
      p_noise = pixel_noise(generator);
      selecter1 = picker(generator);
      selecter2 = picker(generator);
      // get local pixel noise                        this has its own parameter
      // decide if I want to move it and how much     this has its own parameter
      // set pixel (colval, maybe changed row val, pixel value)
      
      

      // so i do this for all of them
      // use Kazus function to turn image2d to numpy array
      // return orig and manip w label

      for (int irow=0; irow<meta.rows(); irow++) {
	for (int icol=0; icol<meta.cols(); icol++) {
	  float newval = img.pixel(irow,icol)*rand.Uniform(); // generate a random value based on the old image value
	  newimg.set_pixel( irow, icol, newval ); // set the value int he new image
	  if ( img.pixel(irow,icol)>0 && newval/img.pixel(irow,icol)>0.5 ) {
	    // if it's above 0.5, we make a Pixel2D object
	    larcv::Pixel2D hit( icol, irow );
	    hit.Intensity( newval );
	    hit.Width( 1.0 );
	    // store the pixel 2D in its output container
	    event_pixel2d->Emplace( (larcv::PlaneID_t)meta.plane(), std::move(hit) );
	  }
	}
      }
      // put the new image into the image output container
      output_container.emplace_back( std::move(newimg) ); // we use emplace and move so that we avoid making a copy
    }
    }





}


  


}
