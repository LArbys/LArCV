#ifndef __DATUMFILLERBASE_CXX__
#define __DATUMFILLERBASE_CXX__

#include "DatumFillerBase.h"
#include "DataFormat/EventImage2D.h"
#include <sstream>

namespace larcv {

  DatumFillerBase::DatumFillerBase(const std::string name)
    : ProcessBase(name)
    , _nentries(kINVALID_SIZE)
    , _image_producer_id  (kINVALID_PRODUCER)
    , _label_producer_id  (kINVALID_PRODUCER)
    , _weight_producer_id (kINVALID_PRODUCER)
    , _multiplicity_producer_id (kINVALID_PRODUCER)
  {}

  const std::string& 
  DatumFillerBase::producer(DatumFillerBase::FillerDataType_t dtype) const
  { switch(dtype) {
    case kFillerImageData: 
      return _image_producer;
    case kFillerLabelData:
      return _label_producer;
    case kFillerWeightData:
      return _weight_producer;
    case kFillerMultiplicityData:
      return _multiplicity_producer;
    }
    return _weight_producer;
  }

  size_t DatumFillerBase::entries() const { return _nentries; }

  bool DatumFillerBase::is(const std::string question) const
  { return (question == "DatumFiller"); }
  
  const std::vector<std::vector<larcv::ImageMeta> >& DatumFillerBase::meta() const
  { return _meta_data; }
  
  const std::vector<float>& 
  DatumFillerBase::data(DatumFillerBase::FillerDataType_t dtype) const
  { switch(dtype) {
    case kFillerImageData: 
      return _image_data;
    case kFillerLabelData:
      return _label_data;
    case kFillerWeightData:
      return _weight_data;
    case kFillerMultiplicityData:
      return _multiplicity_data;
    }
    return _weight_data;
  }

  const std::vector<float>& 
  DatumFillerBase::entry_data(DatumFillerBase::FillerDataType_t dtype) const
  { 
    switch(dtype) {
    case kFillerImageData:
      return _entry_image_data;
    case kFillerLabelData:
      return _entry_label_data;
    case kFillerWeightData:
      return _entry_weight_data;
    case kFillerMultiplicityData:
      return _entry_multiplicity_data;
    }
    return _entry_weight_data;
  }
  
  void DatumFillerBase::configure(const PSet& cfg)
  {
    _image_producer        = cfg.get<std::string>("ImageProducer");
    _label_producer        = cfg.get<std::string>("LabelProducer");
    _weight_producer       = cfg.get<std::string>("WeightProducer","");
    _multiplicity_producer = cfg.get<std::string>("MultiplicityProducer","");
    this->child_configure(cfg);
  }

  void DatumFillerBase::initialize()
  {
    if(_nentries == kINVALID_SIZE) {
      LARCV_CRITICAL() << "# entries not set... must be set @ initialize! " << std::endl;
      throw larbys();
    }
    _current_entry = kINVALID_SIZE;
    _entry_image_size   = _entry_label_size = kINVALID_SIZE;
    _entry_weight_size  = 0;
    _entry_multiplicity_size  = 0;
    _image_producer_id        = kINVALID_PRODUCER;
    _label_producer_id        = kINVALID_PRODUCER;
    _weight_producer_id       = kINVALID_PRODUCER;
    _multiplicity_producer_id = kINVALID_PRODUCER;
    this->child_initialize();
  }

  void DatumFillerBase::batch_begin() {
    if(_entry_image_size != kINVALID_SIZE) {
      _image_data.resize(_nentries * _entry_image_size);
      _label_data.resize(_nentries * _entry_label_size);
      if(!_weight_producer.empty()) 
	_weight_data.resize(_nentries * _entry_weight_size);
      if(!_multiplicity_producer.empty())
	_multiplicity_data.resize(_nentries * _entry_multiplicity_size);
    }
    for( auto& v : _image_data         ) v=0.;
    for( auto& v : _label_data         ) v=0.;
    for( auto& v : _multiplicity_data  ) v=0.;
    for( auto& v : _weight_data        ) v=0.;
    _current_entry = 0;
    _meta_data.clear();
    this->child_batch_begin();
  }

  bool DatumFillerBase::process(IOManager& mgr)
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

      _label_producer_id = mgr.producer_id(_label_product_type,_label_producer);
      if(_label_producer_id == kINVALID_PRODUCER) {
        LARCV_CRITICAL() << "Label producer " << _label_producer << " not valid!" << std::endl;
        throw larbys();
      }
      LARCV_INFO() << std::endl;
      
      if(!_multiplicity_producer.empty()) {
	_multiplicity_producer_id = mgr.producer_id(_multiplicity_product_type,_multiplicity_producer);
	if(_multiplicity_producer_id == kINVALID_PRODUCER) {
	  LARCV_CRITICAL() << "Multiplicity producer " << _multiplicity_producer << " not valid!" << std::endl;
	  throw larbys();
	}
	LARCV_INFO() << std::endl;
      }

      if(!_weight_producer.empty()) {
	_weight_producer_id = mgr.producer_id(_weight_product_type,_weight_producer);
	if(_weight_producer_id == kINVALID_PRODUCER) {
	  LARCV_CRITICAL() << "Weight producer " << _weight_producer << " not valid!" << std::endl;
	  throw larbys();
	}
	LARCV_INFO() << std::endl;
      }
    }
    LARCV_INFO() << std::endl;
    auto const image_data        = mgr.get_data(_image_producer_id);
    auto const label_data        = mgr.get_data(_label_producer_id);
    const larcv::EventImage2D* multiplicity_data = NULL;
    LARCV_INFO() << std::endl;
    if(_entry_image_size==kINVALID_SIZE || _nentries == 1) {

      _entry_image_size        = compute_image_size(image_data);
      _entry_label_size        = compute_label_size(label_data);

      LARCV_INFO() << "Recomputed image size: "        << _entry_image_size << std::endl;
      LARCV_INFO() << "Recomputed label size: "        << _entry_label_size << std::endl;
      LARCV_INFO() << "Recomputed multiplicity size: " << _entry_multiplicity_size << std::endl;

      if( _entry_image_size == kINVALID_SIZE || _entry_label_size == kINVALID_SIZE) {
        LARCV_CRITICAL() << "Rows/Cols/NumChannels not set!" << std::endl;
        throw larbys();
      }

      _image_data.resize         (_entry_image_size         * _nentries, 0.);
      _label_data.resize         (_entry_label_size         * _nentries, 0.);

      if(_weight_producer_id != kINVALID_PRODUCER) {
	_entry_weight_size = _entry_image_size;
	_weight_data.resize (_entry_weight_size * _nentries, 0.);
      }
      if(_multiplicity_producer_id!=kINVALID_PRODUCER) {
	multiplicity_data = (larcv::EventImage2D*)mgr.get_data(_multiplicity_producer_id);
	_entry_multiplicity_size = compute_multiplicity_size(multiplicity_data);
	_multiplicity_data.resize  (_entry_multiplicity_size  * _nentries, 0.);
      }

    }

    EventBase* weight_data=nullptr;
    if(_weight_producer_id != kINVALID_PRODUCER)
      weight_data = mgr.get_data(_weight_producer_id);

    _entry_meta_data.clear();
    this->fill_entry_data(image_data,label_data,weight_data,multiplicity_data);
    
    auto const& entry_image_data        = entry_data(kFillerImageData);
    auto const& entry_label_data        = entry_data(kFillerLabelData);
    auto const& entry_weight_data       = entry_data(kFillerWeightData);
    auto const& entry_multiplicity_data = entry_data(kFillerMultiplicityData);

    if(entry_image_data.size() != _entry_image_size) {
      LARCV_CRITICAL() << "(run,subrun,event) = ("
      << image_data->run() << "," << image_data->subrun() << "," << image_data->event() << ")"
      << "Entry image size should be " << _entry_image_size << " but found " << entry_image_data.size() << std::endl;
      throw larbys();
    }

    if(entry_label_data.size() != _entry_label_size) {
      LARCV_CRITICAL() << "(run,subrun,event) = ("
      << label_data->run() << "," << label_data->subrun() << "," << label_data->event() << ")"
      << "Entry label size should be " << _entry_label_size << " but found " << entry_label_data.size() << std::endl;
      throw larbys();
    }

    if(entry_multiplicity_data.size() != _entry_multiplicity_size) {
      LARCV_CRITICAL() << "(run,subrun,event) = ("
      << multiplicity_data->run() << "," << multiplicity_data->subrun() << "," << multiplicity_data->event() << ")"
      << "Entry multiplicity size should be " << _entry_multiplicity_size << " but found " << entry_multiplicity_data.size() << std::endl;
      throw larbys();
    }


    if(entry_weight_data.size() != _entry_weight_size) {
      LARCV_CRITICAL() << "(run,subrun,event) = ("
      << weight_data->run() << "," << weight_data->subrun() << "," << weight_data->event() << ")"
      << "Entry weight size should be " << _entry_weight_size << " but found " << entry_weight_data.size() << std::endl;
      throw larbys();
    }

    const size_t current_image_index_start = _current_entry * _entry_image_size;
    for(size_t i = 0; i<_entry_image_size; ++i)
      _image_data[current_image_index_start + i] = entry_image_data[i];

    const size_t current_label_index_start = _current_entry * _entry_label_size;
    for(size_t i = 0; i<_entry_label_size; ++i)
      _label_data[current_label_index_start + i] = entry_label_data[i];

    const size_t current_multiplicity_index_start = _current_entry * _entry_multiplicity_size;
    for(size_t i = 0; i<_entry_multiplicity_size; ++i)
      _multiplicity_data[current_multiplicity_index_start + i] = entry_multiplicity_data[i];

    const size_t current_weight_index_start = _current_entry * _entry_weight_size;
    for(size_t i = 0; i<_entry_weight_size; ++i)
      _weight_data[current_weight_index_start + i] = entry_weight_data[i];

    _meta_data.emplace_back(std::move(_entry_meta_data));

    ++_current_entry;
    return true;
  }

  void DatumFillerBase::batch_end()
  { 
    this->child_batch_end();
  }

  void DatumFillerBase::finalize()
  { this->child_finalize(); }

}
#endif
