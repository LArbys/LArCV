#ifndef __IMAGEFROMPIXEL2D_CXX__
#define __IMAGEFROMPIXEL2D_CXX__

#include "ImageFromPixel2D.h"
#include "DataFormat/EventPixel2D.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static ImageFromPixel2DProcessFactory __global_ImageFromPixel2DProcessFactory__;

  ImageFromPixel2D::ImageFromPixel2D(const std::string name)
    : ProcessBase(name)
  {}
    
  void ImageFromPixel2D::configure(const PSet& cfg)
  {
    _pixel_producer  = cfg.get<std::string>("PixelProducer");
    _image_producer  = cfg.get<std::string>("ImageProducer");
    _output_producer = cfg.get<std::string>("OutputProducer");
  }

  void ImageFromPixel2D::initialize()
  {}

  bool ImageFromPixel2D::process(IOManager& mgr)
  {
    auto ev_pixel  = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pixel_producer));
    auto ev_image  = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));
    auto ev_out_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_output_producer));

    auto const& pcluster_m = ev_pixel->Pixel2DClusterArray();
    auto const& image_v = ev_image->Image2DArray();

    for(size_t plane=0; plane<image_v.size(); plane++) {
      auto const& meta = image_v[plane].meta();
      auto iter = pcluster_m.find(plane);
      if(iter==pcluster_m.end()) throw std::exception();
      auto const& pcluster_v = (*iter).second;
      Image2D out_image(meta);
      out_image.paint(0.);
      for(auto const& pcluster : pcluster_v) {
	for(auto const& pixel2d : pcluster) {
	  //std::cout<<pixel2d.X() << " " << pixel2d.Y() << " ... " << pixel2d.Intensity() << std::endl;
	  //out_image.set_pixel(pixel2d.Y(), pixel2d.X(),100);
	  out_image.set_pixel( (pixel2d.X() * meta.rows() + pixel2d.Y()),100);
	}
      }
      ev_out_image->Emplace(std::move(out_image));
    }
    return true;
  }

  void ImageFromPixel2D::finalize()
  {}

}
#endif
