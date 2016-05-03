#ifndef __BINARIZE_CXX__
#define __BINARIZE_CXX__

#include "Binarize.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static BinarizeProcessFactory __global_BinarizeProcessFactory__;

  Binarize::Binarize(const std::string name)
    : ProcessBase(name)
  {

  }
    
  void Binarize::configure(const PSet& cfg)
  {
    fChannelThresholds   = cfg.get<std::vector<float> >( "ChannelThresholds" );
    fInputImageProducer  = cfg.get<std::string>( "InputImageProducer" );
    fOutputImageProducer = cfg.get<std::string>( "OutputImageProducer" );
  }

  void Binarize::initialize()
  {}

  bool Binarize::process(IOManager& mgr)
  {
    auto input_imgs  = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,fInputImageProducer));
    auto output_imgs = (larcv::EventImage2D*)(mgr.get_data(kProductImage2D,fOutputImageProducer));
    for ( size_t ch=0; ch< input_imgs->Image2DArray().size(); ch++ ) {
      larcv::Image2D img( input_imgs->Image2DArray().at(ch) ); // copy
      float thresh = fChannelThresholds.at(ch);
      for ( size_t r=0; r<img.meta().rows(); r++ ) {
	for ( size_t c=0; c<img.meta().cols(); c++) {
	  float adc = img.pixel( r, c );
	  if ( adc>thresh ) {
	    img.set_pixel( r, c, 1.0 );
	  }
	  else {
	    img.set_pixel( r, c, -1.0 );
	  }
	}
      }
      
      output_imgs->Emplace( std::move(img) );

    }
    return true;
  }

  void Binarize::finalize()
  {}

}
#endif
