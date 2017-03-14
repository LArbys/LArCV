#ifndef __IMAGEFROMPIXEL2D_CXX__
#define __IMAGEFROMPIXEL2D_CXX__

#include "ImageFromPixel2D.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPixel2D.h"
namespace larcv {

  static ImageFromPixel2DProcessFactory __global_ImageFromPixel2DProcessFactory__;

  ImageFromPixel2D::ImageFromPixel2D(const std::string name)
    : ProcessBase(name)
  {}
    
  void ImageFromPixel2D::configure(const PSet& cfg)
  {
    _pixel2d_producer = cfg.get<std::string>("Pixel2DProducer");
    _ref_image_producer = cfg.get<std::string>("RefImageProducer");
    _out_image_producer = cfg.get<std::string>("OutImageProducer");

    if(_out_image_producer == _ref_image_producer) {
      LARCV_CRITICAL() << "Not supporting in-memory modification yet..." << std::endl;
      throw larbys();
    }
  }

  void ImageFromPixel2D::initialize()
  {}

  bool ImageFromPixel2D::process(IOManager& mgr)
  {
    auto ev_pixel2d = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pixel2d_producer));
    if(!ev_pixel2d) {
      LARCV_CRITICAL() << "Pixel2D by producer " << _pixel2d_producer << " not found!" << std::endl;
      throw larbys();
    }
    
    auto ev_ref_image2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_ref_image_producer));
    if(!ev_ref_image2d) {
      LARCV_CRITICAL() << "Image2D by producer " << _ref_image_producer << " not found!" << std::endl;
      throw larbys();
    }

    auto ev_out_image2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_out_image_producer));
    if(!ev_out_image2d) {
      LARCV_CRITICAL() << "Image2D by producer " << _out_image_producer << " could not be created!" << std::endl;
      throw larbys();
    }

    auto const& image_v = ev_ref_image2d->Image2DArray();
    std::vector<larcv::Image2D> out_image_v;
    for(auto const& img : image_v) {
      out_image_v.push_back(img);
      out_image_v.back().paint(0);
    }
    
    auto const& cluster_m = ev_pixel2d->Pixel2DClusterArray();
    std::cout<<cluster_m.size()<<std::endl;
    for(auto const& id_cluster : cluster_m) {

      auto const& image_idx = id_cluster.first;
      auto const& cluster_v = id_cluster.second;
      size_t tot_npx = 0;
      for(auto const& cluster : cluster_v)
	tot_npx += cluster.size();

      if(tot_npx<1) continue;

      /*
      for(auto const& cluster : cluster_v) {
	for(auto const& pt : cluster)
	  std::cout << image_idx<<","<<pt.X() << "," << pt.Y() << "," << pt.Intensity() << std::endl;
      }
      std::cout<<std::endl;
      */
      
      if(image_idx >= out_image_v.size()) {
	LARCV_CRITICAL() << "Image index " << image_idx << " not available... " << std::endl;
	throw larbys();
      }
      
      auto& img = out_image_v[image_idx];
      for(auto const& cluster: cluster_v) {
	for(auto const& px : cluster) {
	  if(px.X() >= img.meta().cols()){
	    LARCV_WARNING() << "Skipping pixel (" << px.X() << "," << px.Y() << ")"
			    << " as it is outside the X boundary (" << img.meta().cols() << ") ..." << std::endl;
	    continue;
	  }
	  if(px.Y() >= img.meta().rows()){
	    LARCV_WARNING() << "Skipping pixel (" << px.X() << "," << px.Y() << ")"
			    << " as it is outside the Y boundary (" << img.meta().rows() << ") ..." << std::endl;
	    continue;
	  }
	  size_t row = (size_t)(img.meta().rows()-px.Y()-1);
	  size_t col = (size_t)(px.X());
	  float  val = (float)(px.Intensity());
	  if(img.pixel(row,col) < val) img.set_pixel(row,col,val);
	}
      }
    }

    ev_out_image2d->Emplace(std::move(out_image_v));
    return true;
  }

  void ImageFromPixel2D::finalize()
  {}

}
#endif
