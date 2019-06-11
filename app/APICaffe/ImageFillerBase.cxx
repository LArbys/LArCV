#ifndef __IMAGEFILLERBASE_CXX__
#define __IMAGEFILLERBASE_CXX__

#include "ImageFillerBase.h"
#include "DataFormat/EventImage2D.h"
#include <sstream>

namespace larcv {

  ImageFillerBase::ImageFillerBase(const std::string name)
    : ProcessBase(name)
    , _nentries(kINVALID_SIZE)
    , _image_producer_id  (kINVALID_PRODUCER)
  {}

  const std::string& 
  ImageFillerBase::producer(ImageFillerBase::FillerDataType_t dtype) const
  { switch(dtype) {
    case kFillerImageData: 
      return _image_producer;
    }
    return _image_producer;
  }

  size_t ImageFillerBase::entries() const { return _nentries; }

  bool ImageFillerBase::is(const std::string question) const
  { return (question == "ImageFiller"); }
  
  const std::vector<std::vector<larcv::ImageMeta> >& ImageFillerBase::meta() const
  { return _meta_data; }
  
  const std::vector<float>& 
  ImageFillerBase::data(ImageFillerBase::FillerDataType_t dtype) const
  { switch(dtype) {
    case kFillerImageData: 
      return _image_data;
    }
    return _image_data;
  }

  const std::vector<float>& 
  ImageFillerBase::entry_data(ImageFillerBase::FillerDataType_t dtype) const
  { 
    switch(dtype) {
    case kFillerImageData:
      return _entry_image_data;
    }
    return _entry_image_data;
  }
  
  void ImageFillerBase::configure(const PSet& cfg)
  {
    _image_producer        = cfg.get<std::string>("ImageProducer");
    this->child_configure(cfg);
  }

  void ImageFillerBase::initialize()
  {
    if(_nentries == kINVALID_SIZE) {
      LARCV_CRITICAL() << "# entries not set... must be set @ initialize! " << std::endl;
      throw larbys();
    }
    _current_entry = kINVALID_SIZE;
    _entry_image_size = kINVALID_SIZE;
    _image_producer_id        = kINVALID_PRODUCER;
    this->child_initialize();
  }

  void ImageFillerBase::batch_begin() {
    if(_entry_image_size != kINVALID_SIZE) {
      _image_data.resize(_nentries * _entry_image_size);
    }
    for( auto& v : _image_data         ) v=0.;
    _current_entry = 0;
    _meta_data.clear();
    this->child_batch_begin();
  }

  bool ImageFillerBase::process(IOManager& mgr)
  {
    LARCV_INFO() << std::endl;

    if(_current_entry == kINVALID_SIZE) {
      LARCV_CRITICAL() << "batch_begin() not called... Note this process is only meant to use with ThreadReadDriver!\n";
      throw larbys();
    }
    LARCV_INFO() << std::endl;
    if(_image_producer_id == kINVALID_PRODUCER) {

      _image_producer_id = mgr.producer_id(_image_product_type,_image_producer);
      if(_image_producer_id == kINVALID_PRODUCER) {
        LARCV_CRITICAL() << "Image producer " << _image_producer << " not valid!" << std::endl;
        throw larbys();
      }
    }
    LARCV_INFO() << std::endl;
    auto const image_data        = mgr.get_data(_image_producer_id);
    LARCV_INFO() << std::endl;
    if(_entry_image_size==kINVALID_SIZE || _nentries == 1) {

      _entry_image_size        = compute_image_size(image_data);

      LARCV_INFO() << "Recomputed image size: "        << _entry_image_size << std::endl;

      if( _entry_image_size == kINVALID_SIZE ) {
        LARCV_CRITICAL() << "Rows/Cols/NumChannels not set!" << std::endl;
        throw larbys();
      }

      _image_data.resize         (_entry_image_size         * _nentries, 0.);

    }

    _entry_meta_data.clear();
    this->fill_entry_data(image_data);
    
    auto const& entry_image_data        = entry_data(kFillerImageData);

    if(entry_image_data.size() != _entry_image_size) {
      LARCV_CRITICAL() << "(run,subrun,event) = ("
      << image_data->run() << "," << image_data->subrun() << "," << image_data->event() << ")"
      << "Entry image size should be " << _entry_image_size << " but found " << entry_image_data.size() << std::endl;
      throw larbys();
    }

    const size_t current_image_index_start = _current_entry * _entry_image_size;
    for(size_t i = 0; i<_entry_image_size; ++i)
      _image_data[current_image_index_start + i] = entry_image_data[i];

    _meta_data.emplace_back(std::move(_entry_meta_data));

    ++_current_entry;
    return true;
  }

  void ImageFillerBase::batch_end()
  { 
    this->child_batch_end();
  }

  void ImageFillerBase::finalize()
  { this->child_finalize(); }

}
#endif
