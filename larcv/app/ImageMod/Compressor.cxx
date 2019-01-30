#ifndef __COMPRESSOR_CXX__
#define __COMPRESSOR_CXX__

#include "Compressor.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static CompressorProcessFactory __global_CompressorProcessFactory__;

  Compressor::Compressor(const std::string name)
    : ProcessBase(name)
  {}
    
  void Compressor::configure(const PSet& cfg)
  {
    _image_producer_v  = cfg.get<std::vector<std::string> >("ImageProducer");
    _row_compression_v = cfg.get<std::vector<size_t     > >("RowCompression");
    _col_compression_v = cfg.get<std::vector<size_t     > >("ColCompression");

    auto mode_v = cfg.get<std::vector<unsigned short> >("Mode");
    _mode_v.clear();
    for(auto const& v : mode_v) _mode_v.push_back( (Image2D::CompressionModes_t)v );

    if(_mode_v.size() != _image_producer_v.size()  ||
       _mode_v.size() != _row_compression_v.size() ||
       _mode_v.size() != _col_compression_v.size() ) {
      LARCV_CRITICAL() << "Length of parameter arrays do not match!" << std::endl;
      throw larbys();
    }
  }

  void Compressor::initialize()
  {
    for(size_t i=0; i<_mode_v.size(); ++i) {

      auto const& image_producer  = _image_producer_v[i];
      auto const& row_compression = _row_compression_v[i];
      auto const& col_compression = _col_compression_v[i];

      if(!row_compression) {
	LARCV_CRITICAL() << "Row compression factor is 0 (undefined)!" << std::endl;
	throw larbys();
      }
      if(!col_compression) {
	LARCV_CRITICAL() << "Col compression factor is 0 (undefined)!" << std::endl;
	throw larbys();
      }
      if(image_producer.empty()) {
	LARCV_CRITICAL() << "Image producer not specified!" << std::endl;
	throw larbys();
      }
    }
  }

  bool Compressor::process(IOManager& mgr)
  {
    // Check if compression factor works
    for(size_t i=0; i<_mode_v.size(); ++i) {
      
      auto const& image_producer  = _image_producer_v[i];
      auto const& row_compression = _row_compression_v[i];
      auto const& col_compression = _col_compression_v[i];
      
      auto ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,image_producer));
      if(!ev_image) {
	LARCV_CRITICAL() << "Input image not found by producer name " << image_producer << std::endl;
	throw larbys();
      }

      for(auto const& img : ev_image->Image2DArray()) {
	auto const& meta = img.meta();
	if(meta.rows() % row_compression) {
	  LARCV_CRITICAL() << "Input image # rows (" << meta.rows()
			   << ") cannot be divide by compression factor (" << row_compression
			   << ")" << std::endl;
	  throw larbys();
	}
	if(meta.cols() % col_compression) {
	  LARCV_CRITICAL() << "Input image # cols (" << meta.cols()
			   << ") cannot be divide by compression factor (" << col_compression
			   << ")" << std::endl;
	  throw larbys();
	}
      }
    }
    
    // Apply compression
    for(size_t i=0; i<_mode_v.size(); ++i) {
      
      auto const& image_producer  = _image_producer_v[i];
      auto const& row_compression = _row_compression_v[i];
      auto const& col_compression = _col_compression_v[i];
      auto const& mode = _mode_v[i];
      auto ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,image_producer));
      
      std::vector<larcv::Image2D> image_v;
      ev_image->Move(image_v);
      for(auto& img : image_v) {
	
	img.compress( img.meta().rows() / row_compression,
		      img.meta().cols() / col_compression,
		      mode );
	
      }
      ev_image->Emplace(std::move(image_v));
    }
    
    return true;
  }
  

  void Compressor::finalize()
  {}

}
#endif
