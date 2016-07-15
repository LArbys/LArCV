#ifndef __SEGDATUMFILLERBASE_CXX__
#define __SEGDATUMFILLERBASE_CXX__

#include "SegDatumFillerBase.h"
#include "DataFormat/EventImage2D.h"
#include <sstream>

namespace larcv {

  SegDatumFillerBase::SegDatumFillerBase(const std::string name)
    : ProcessBase(name)
    , _nentries(kINVALID_SIZE)
    , _image_producer_id(kINVALID_PRODUCER)
    , _seg_image_producer_id(kINVALID_PRODUCER)
  {}
    
  bool SegDatumFillerBase::is(const std::string& question) const
  { return (question == "DatumFiller"); }

  void SegDatumFillerBase::configure(const PSet& cfg)
  {
    _image_producer = cfg.get<std::string>("InputProducer");
    _seg_image_producer = cfg.get<std::string>("SegProducer");
    this->child_configure(cfg);
  }

  void SegDatumFillerBase::initialize()
  {

    if(_nentries == kINVALID_SIZE) {
      LARCV_CRITICAL() << "# entries not set... must be set @ initialize! " << std::endl;
      throw larbys();
    }
    _entry_image_size = _entry_label_size = kINVALID_SIZE;
    _current_entry = kINVALID_SIZE;
    _image_producer_id = kINVALID_PRODUCER;
    _label_producer_id = kINVALID_PRODUCER;

    this->child_initialize();
  }

  void SegDatumFillerBase::batch_begin() {

    if(_entry_image_size != kINVALID_SIZE)
      _image_data.resize(_nentries * _entry_image_size);

    if(_entry_label_size != kINVALID_SIZE)
      _label_data.resize(_nentries * _entry_label_size);

    for(auto& v : _image_data) v=0;
    for(auto& v : _label_data) v=0.;
    _current_entry = 0;
    this->child_batch_begin();
  }

  bool SegDatumFillerBase::process(IOManager& mgr)
  {
    if(_current_entry == kINVALID_SIZE) {
      LARCV_CRITICAL() << "batch_begin() not called... Note this process is only meant to use with ThreadReadDriver!\n";
      throw larbys();
    }

    if(_image_producer_id == kINVALID_PRODUCER) {

      _image_producer_id = mgr.producer_id(kProductImage2D,_image_producer);
      if(_image_producer_id == kINVALID_PRODUCER) {
        LARCV_CRITICAL() << "Image producer " << _image_producer << " not valid!" << std::endl;
        throw larbys();
      }

      _label_producer_id = mgr.producer_id(kProductImage2D,_label_producer);
      if(_label_producer_id == kINVALID_PRODUCER) {
        LARCV_CRITICAL() << "Label producer " << _label_producer << " not valid!" << std::endl;
        throw larbys();
      }
          
    }

    auto const event_image_data = mgr.get_data(_image_producer_id);

    auto const event_label_data = mgr.get_data(_seg_image_producer_id);

    if(_entry_image_size == kINVALID_SIZE) {

      set_dimension(event_image_data, event_label_data);

      if(_entry_image_size == kINVALID_SIZE)
        LARCV_CRITICAL() << "Image data size per entry is not set!" << std::endl;
        throw larbys();
      }
      else LARCV_INFO() << "Image data size per entry: " << _event_image_size << std::endl;

      if(_entry_label_size == kINVALID_SIZE)
        LARCV_CRITICAL() << "Label data size per entry is not set!" << std::endl;
        throw larbys();
      }
      else LARCV_INFO() << "Label data size per entry: " << _label_image_size << std::endl;

      _data.resize(_nentries * _entry_image_size,0.);
      _labels.resize(_nentries * _entry_label_size,0.);
    }
    
    this->fill_entry_data(event_image_data, event_label_data);

    auto const& entry_image_data = entry_image();
    auto const& entry_label_data = entry_label();
    
    if(entry_image_data.size() != _entry_data_size) {
      LARCV_CRITICAL() << "(run,subrun,event) = ("
      << event_image->run() << "," << event_image->subrun() << "," << event_image->event() << ")"
      << " fill_entry_data() filled IMAGE data with size " << entry_image_data.size()
      << " where it is supposed to be " << _entry_image_size << std::endl;
      throw larbys();
    }

    const size_t current_image_index_start = _current_entry * _entry_image_size;
    for(size_t i = 0; i<_entry_data_size; ++i)
      _image_data[current_image_index_start + i] = entry_image_data[i]

    const size_t current_label_index_start = _current_entry * _entry_label_size;
    for(size_t i = 0; i<_entry_label_size; ++i)
      _label_data[current_label_index_start + i] = entry_label_data[i]

    ++_current_entry;
    return true;
  }

  void SegDatumFillerBase::batch_end()
  { 

    this->child_batch_end(); 
  }

  void SegDatumFillerBase::finalize()
  { this->child_finalize(); }

}
#endif
