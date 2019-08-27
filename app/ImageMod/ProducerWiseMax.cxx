#ifndef __PRODUCER_WISE_MAX_CXX__
#define __PRODUCER_WISE_MAX_CXX__

#include <sstream>

#include "ProducerWiseMax.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static ProducerWiseMaxProcessFactory __global_ProducerWiseMaxProcessFactory__;

  ProducerWiseMax::ProducerWiseMax(const std::string name)
    : ProcessBase(name)
  {}
    
  void ProducerWiseMax::configure(const PSet& cfg)
  {
    _in_producer_v    = cfg.get< std::vector<std::string> >("InProducer");
    _out_producer     = cfg.get<std::string>("OutputProducer");
    _nproducers       = _in_producer_v.size(); 

    _producer_weight_v   = cfg.get<std::vector<float> >("ProducerWeights",{});
    if (_producer_weight_v.empty()) {
      _producer_weight_v.resize(_in_producer_v.size(),1.0);
    }

    auto producer_mask_v = cfg.get<std::vector<float> >("ProducerMask",{});
    _producer_mask_v.resize(_nproducers);
    if (producer_mask_v.empty()) {
      for(size_t plane=0;plane<_nproducers;++plane) _producer_mask_v[plane] = (float)plane;
    } else {
      _producer_mask_v = producer_mask_v;
    }
  }

  void ProducerWiseMax::initialize()
  {}

  bool ProducerWiseMax::process(IOManager& mgr)
  {

    std::vector< EventImage2D* > _evimg_v;
    for ( size_t p=0; p<_nproducers; p++ ) {
      auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_in_producer_v[p]));
      if(!event_image) {
	std::stringstream errmsg;
	errmsg << "No event image found for producer[" << p << "]: " << _in_producer_v[p] << std::endl;
	throw larbys(errmsg.str());
      }
      LARCV_DEBUG() << "  planes in [" << _in_producer_v[p] << "]: " << event_image->Image2DArray().size() << std::endl;
      if ( event_image->Image2DArray().size()==0 )
	throw larbys("  no images in event container!");
      _evimg_v.push_back( event_image );
    }
    LARCV_DEBUG() << "Number of producers: " << _evimg_v.size() << std::endl;
      
    if (_evimg_v.empty()) return false;

    // for now, images compared must have same number of planes and same metas
    // else things get complicated
    int nplanes = -1;
    std::vector<larcv::ImageMeta> plane_metas;
    for ( size_t p=0; p<_nproducers; p++ ) {
      
      // check number of planes
      auto const& img_v = _evimg_v[p]->Image2DArray();
      if ( nplanes<0 )
	nplanes = (int)img_v.size();
      else if ( nplanes!=(int)img_v.size() ) {
	throw larbys("Number of planes across input Image2D producers must be the same");
      }

      if ( p==0 ) {
	for ( auto const& img : img_v )
	  plane_metas.push_back( img.meta() );
      }
      else {
	for ( size_t ip=0; ip<nplanes; ip++ ) {
	  if ( img_v[ip].meta()!=plane_metas[ip] ) {
	    throw larbys("Image metas do not match");
	  }
	}
      }
    }
    LARCV_DEBUG() << "Number of planes: " << nplanes << std::endl;

    // make output images
    std::vector< larcv::Image2D> out_v;
    for ( auto const& meta : plane_metas ){
      larcv::Image2D image(meta);
      image.paint(0);
      out_v.emplace_back( std::move(image) );
      LARCV_DEBUG() << "meta of plane[" << meta.plane() << "]: " << meta.dump() << std::endl;
    }

    for ( size_t plane=0; plane<nplanes; plane++ ) {
      auto& image = out_v[plane];

      for(size_t row=0;row<image.meta().rows();++row){
	for(size_t col=0;col<image.meta().cols();++col){

	  float maxpx(-1),maxpl(-1);
	  for(size_t p=0;p<_nproducers;++p) {
	    auto px=_evimg_v[p]->Image2DArray().at(plane).pixel(row,col);
	    px*=_producer_weight_v[p];
	    if (px>maxpx) { maxpx=px; maxpl=p; }
	  }
	  if (maxpx<0 or maxpl<0) throw larbys("No max producer identified");
	  image.set_pixel(row,col,_producer_mask_v[maxpl]);
	}
      }
    }
    
    LARCV_DEBUG() << "store in output producer: " << _out_producer << std::endl;
    auto out_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_out_producer));
    out_image->Emplace(std::move(out_v));

    return true;
  }

  void ProducerWiseMax::finalize()
  {}

}
#endif
