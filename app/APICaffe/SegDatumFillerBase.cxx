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
  { return (question == "SegDatumFiller"); }

  void SegDatumFillerBase::configure(const PSet& cfg)
  {
    std::string mean_image_producer="";
    std::string mean_image_fname="";
    _mean_adc_v.clear();
    _mean_adc_v = cfg.get<std::vector<float> >("MeanADC",_mean_adc_v);
    mean_image_fname = cfg.get<std::string>("MeanImageFile",mean_image_fname);
    mean_image_producer = cfg.get<std::string>("MeanImageProducer",mean_image_producer);
    if(!mean_image_fname.empty()) {
      LARCV_NORMAL() << "Retrieving a mean image from " << mean_image_fname 
        << " ... producer " << mean_image_producer << std::endl;
      IOManager mean_io(IOManager::kREAD);
      mean_io.add_in_file(mean_image_fname);
      mean_io.initialize();
      mean_io.read_entry(0);
      auto const event_image = (EventImage2D*)(mean_io.get_data(kProductImage2D,mean_image_producer));
      if(!event_image || event_image->Image2DArray().empty()) {
        LARCV_CRITICAL() << "Failed to retrieve mean image array!" << std::endl;
        throw larbys();
      }
      _mean_image_v = event_image->Image2DArray();
      mean_io.finalize();
    }
    if(_mean_adc_v.empty() && _mean_image_v.empty()) {
      LARCV_CRITICAL() << "Both mean image and adc values are empty!" << std::endl;
      throw larbys();
    }

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

    _num_channels = _rows = _cols = kINVALID_SIZE;
    _label = kINVALID_FLOAT;
    _current_entry = kINVALID_SIZE;
    _entry_data_size = 0;
    _image_producer_id = kINVALID_PRODUCER;
    _seg_image_producer_id = kINVALID_PRODUCER;

    this->child_initialize();
  }

  void SegDatumFillerBase::batch_begin() {

    if(_rows != kINVALID_SIZE) {
      _data.resize(_nentries * _rows * _cols * _num_channels);
      _labels.resize(_nentries * _rows * _cols * _num_channels);
    }

    for(auto& v : _data)   v=0;
    for(auto& v : _labels) v=0.;
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

      _seg_image_producer_id = mgr.producer_id(kProductImage2D,_seg_image_producer);
      if(_seg_image_producer_id == kINVALID_PRODUCER) {
        LARCV_CRITICAL() << "ROI producer " << _seg_image_producer << " not valid!" << std::endl;
        throw larbys();
      }

      
      // LArCV file may not have segmentation producer, for now it's a 
      // requirement that will be lifted once i get shit working
      _seg_image_producer_id = mgr.producer_id(kProductImage2D,_seg_image_producer);
      if(_seg_image_producer_id == kINVALID_PRODUCER) {
        LARCV_CRITICAL() << "Seg Image producer " << _seg_image_producer << " not valid!" << std::endl;
        throw larbys();
      }
          
    }

    auto const event_image   = (EventImage2D*)(mgr.get_data(_image_producer_id));

    //shouldn't be a requirement
    auto const segment_image = (EventImage2D*)(mgr.get_data(_seg_image_producer_id));

    auto const& image_v = event_image->Image2DArray();
    auto const& seg_image_v  = segment_image->Image2DArray();

    if(_rows==kINVALID_SIZE) {

      set_dimension(image_v);

      if(_rows==kINVALID_SIZE || _cols==kINVALID_SIZE || _num_channels==kINVALID_SIZE) {
        LARCV_CRITICAL() << "Rows/Cols/NumChannels not set!" << std::endl;
        throw larbys();
      }
      if(_rows==0 || _cols==0 || _num_channels==0) {
        LARCV_CRITICAL() << "Rows/Cols/NumChannels (either one or more) is 0!" << std::endl;
        throw larbys();
      }
      _entry_data_size = _rows * _cols * _num_channels;
      _data.resize(_nentries * _entry_data_size,0.);
      _labels.resize(_nentries * _entry_data_size,0.);
    }
    
    
    
    //vic come back to me!
    _label = kINVALID_FLOAT;
    this->fill_entry_data(image_v,seg_image_v);

    auto const& one_data = entry_data();
    auto const& one_seg_data = seg_entry_data();
    
    //shit, should actually check this

    // if(_label == kINVALID_FLOAT) {
    //   LARCV_CRITICAL() << "Label not set!" << std::endl;
    //   throw larbys();
    // }

    //careful here, lets set the labels later on
    //_labels[_current_entry] = _label;

    if(one_data.size() != _entry_data_size) {
      LARCV_CRITICAL() << "(run,subrun,event) = ("
      << event_image->run() << "," << event_image->subrun() << "," << event_image->event() << ")"
      << " fill_data() return is not the right length (" << _entry_data_size
      << " based on (rows,cols,ch) = (" << _rows << "," << _cols << "," << _num_channels << ")).\n";
      throw larbys();
    }

    const size_t current_index_start = _current_entry * _entry_data_size;
    for(size_t i = 0; i<_entry_data_size; ++i) {
      _data[current_index_start + i]  = one_data[i];
      _labels[current_index_start + i] = one_seg_data[i];
    }

    ++_current_entry;
    return true;
  }

  void SegDatumFillerBase::batch_end()
  { 
    if(logger().level() <= msg::kINFO) {
      std::vector<size_t> ctr_v;
      for(auto const& v : _labels) {
	if(v>=ctr_v.size()) ctr_v.resize(v+1,0);
	ctr_v[v] += 1;
      }
      std::stringstream ss;
      ss << "Used: ";
      for(size_t i=0;i<ctr_v.size();++i)
	ss << ctr_v[i] << " of class " << i << " ... ";
      LARCV_INFO() << ss.str() << std::endl;
    }
    this->child_batch_end(); 
  }

  void SegDatumFillerBase::finalize()
  { this->child_finalize(); }

}
#endif
