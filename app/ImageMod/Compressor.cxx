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
    _image_producer  = cfg.get<std::string>("ImageProducer");
    _row_compression = cfg.get<size_t>("RowCompression");
    _col_compression = cfg.get<size_t>("ColCompression");
    _mode            = (Image2D::CompressionModes_t)(cfg.get<unsigned short>("Mode"));
  }

  void Compressor::initialize()
  {
    if(!_row_compression) {
      LARCV_CRITICAL() << "Row compression factor is 0 (undefined)!" << std::endl;
      throw larbys();
    }
    if(!_col_compression) {
      LARCV_CRITICAL() << "Col compression factor is 0 (undefined)!" << std::endl;
      throw larbys();
    }
    if(_image_producer.empty()) {
      LARCV_CRITICAL() << "Image producer not specified!" << std::endl;
      throw larbys();
    }
  }

  bool Compressor::process(IOManager& mgr)
  {
    auto ev_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));
    if(!ev_image) {
      LARCV_CRITICAL() << "Input image not found by producer name " << _image_producer << std::endl;
      throw larbys();
    }
    // Check if compression factor works
    for(auto const& img : ev_image->Image2DArray()) {
      auto const& meta = img.meta();
      if(meta.rows() % _row_compression) {
	LARCV_CRITICAL() << "Input image # rows (" << meta.rows()
			 << ") cannot be divide by compression factor (" << _row_compression
			 << ")" << std::endl;
	throw larbys();
      }
      if(meta.cols() % _col_compression) {
	LARCV_CRITICAL() << "Input image # cols (" << meta.cols()
			 << ") cannot be divide by compression factor (" << _col_compression
			 << ")" << std::endl;
	throw larbys();
      }
    }
    // Apply compression
    std::vector<larcv::Image2D> image_v;
    ev_image->Move(image_v);
    for(auto& img : image_v) {

      img.compress( img.meta().rows() / _row_compression,
		    img.meta().cols() / _col_compression,
		    _mode );

    }
    ev_image->Emplace(std::move(image_v));
    return true;
  }

  void Compressor::finalize()
  {}

}
#endif
