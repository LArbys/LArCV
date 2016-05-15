#ifndef __SIMPLEFILLER_CXX__
#define __SIMPLEFILLER_CXX__

#include "SimpleFiller.h"
#include "DataFormat/UtilFunc.h"
#include <random>
namespace larcv {

  static SimpleFillerProcessFactory __global_SimpleFillerProcessFactory__;

  SimpleFiller::SimpleFiller(const std::string name)
    : DatumFillerBase(name)
    , _slice_v()
    , _max_ch(0)
    , _max_adc_v()
    , _min_adc_v()
  {}

  void SimpleFiller::child_configure(const PSet& cfg)
  {
    _slice_v = cfg.get<std::vector<size_t> >("Channels",_slice_v);
    _max_adc_v = cfg.get<std::vector<float> >("MaxADC");
    _min_adc_v = cfg.get<std::vector<float> >("MinADC");    
    _adc_gaus_mean = cfg.get<double>("GausSmearingMean",1.0);
    _adc_gaus_sigma = cfg.get<double>("GuasSmearingSigma",-1.0);
    _adc_gaus_pixelwise = cfg.get<bool>("PixelWiseSmearing");
    auto type_to_class = cfg.get<std::vector<unsigned short> >("ClassTypeList");
    if(type_to_class.empty()) {
      LARCV_CRITICAL() << "ClassTypeList needed to define classes!" << std::endl;
      throw larbys();
    }
    _roitype_to_class.clear();
    _roitype_to_class.resize(kROITypeMax,kINVALID_SIZE);
    for(size_t i=0; i<type_to_class.size(); ++i) {
      auto const& type = type_to_class[i];
      if(type >= kROITypeMax) {
	LARCV_CRITICAL() << "ClassTypeList contains type " << type << " which is not a valid ROIType_t!" << std::endl;
	throw larbys();
      }
      _roitype_to_class[type] = i;
    }

  }

  void SimpleFiller::child_initialize()
  { _entry_data.clear(); }

  void SimpleFiller::child_batch_begin() {}

  void SimpleFiller::child_batch_end()   {}

  void SimpleFiller::child_finalize()    {}

  void SimpleFiller::set_dimension(const std::vector<larcv::Image2D>& image_v)
  {
    if(image_v.empty()) {
      LARCV_CRITICAL() << "Input image is empty!" << std::endl;
      throw larbys();
    }
    if(_slice_v.empty()) {
      _slice_v.resize(image_v.size());
      for(size_t i=0; i<_slice_v.size(); ++i) _slice_v[i] = i;
    }

    _num_channels = _slice_v.size();
    _max_ch = 0;
    for(auto const& v : _slice_v) if(_max_ch < v) _max_ch = v;

    if(image_v.size() <= _max_ch) {
      LARCV_CRITICAL() << "Requested slice max channel (" << _max_ch 
        << ") exceeds available # of channels in the input image" << std::endl;
      throw larbys();
    }
    _rows = image_v.front().meta().rows();
    _cols = image_v.front().meta().cols();

    // Make sure mean image/adc has right number of channels
    auto const& mean_adc_v = this->mean_adc();
    auto const& mean_image_v = this->mean_image();
    if(mean_adc_v.size() && mean_adc_v.size() != _slice_v.size()) {
      LARCV_CRITICAL() << "Mean adc array dimension do not match with channel size!" << std::endl;
      throw larbys();
    }
    if(mean_image_v.size()) {
      if(mean_image_v.size() != image_v.size()) {
        LARCV_CRITICAL() << "Mean image array dimension do not match with input data image!" << std::endl;
        throw larbys();        
      }
      for(auto const& img : mean_image_v) {
        if(img.meta().rows() != _rows) {
          LARCV_CRITICAL() << "Mean image row count do not match! " << std::endl;
          throw larbys();
        }
        if(img.meta().cols() != _cols) {
          LARCV_CRITICAL() << "Mean image col count do not match! " << std::endl;
          throw larbys();
        }
      }
    }
    // Make sure min/max adc vector size makes sense
    if(_min_adc_v.size() && _min_adc_v.size() != _slice_v.size()) {
      LARCV_CRITICAL() << "Min adc array dimension do not match with channel size!" << std::endl;
      throw larbys();      
    }
    if(_max_adc_v.size() && _max_adc_v.size() != _slice_v.size()) {
      LARCV_CRITICAL() << "Max adc array dimension do not match with channel size!" << std::endl;
      throw larbys();      
    }
    // Define caffe idx to Image2D idx
    _caffe_idx_to_img_idx.resize(_rows*_cols,0);
    size_t caffe_idx = 0;
    for(size_t row=0; row<_rows; ++row) {
      for(size_t col=0; col<_cols; ++col) {
        _caffe_idx_to_img_idx[caffe_idx] = col*_rows + row;
        ++caffe_idx;
      }
    }
  }

  void SimpleFiller::assert_dimension(const std::vector<larcv::Image2D>& image_v)
  {
    if(_rows == kINVALID_SIZE) {
      LARCV_WARNING() << "set_dimension() must be called prior to check_dimension()" << std::endl;
      return;
    }
    bool valid_ch   = (image_v.size() > _max_ch);
    bool valid_rows = true;
    for(auto const& img : image_v) {
      valid_rows = ( _rows == img.meta().rows() );
      if(!valid_rows) break;
    }
    bool valid_cols = true;
    for(auto const& img : image_v) {
    valid_cols = ( _cols == img.meta().cols() );
      if(!valid_cols) break;
    }
    if(!valid_rows) {
      LARCV_CRITICAL() << "# of rows in the input image have changed!" << std::endl;
      throw larbys();
    }
    if(!valid_cols) {
      LARCV_CRITICAL() << "# of cols in the input image have changed!" << std::endl;
      throw larbys();
    }
    if(!valid_ch) {
      LARCV_CRITICAL() << "# of channels have changed in the input image!" << std::endl;
      throw larbys();
    }
  }

  void SimpleFiller::fill_entry_data( const std::vector<larcv::Image2D>& image_v,
                                      const std::vector<larcv::ROI>& roi_v)
  {
    this->assert_dimension(image_v);
    const size_t batch_size = _rows * _cols * _num_channels;
    if(_entry_data.empty()) _entry_data.resize(batch_size,0.);
    for(auto& v : _entry_data) v = 0.;

    auto const& mean_image_v = mean_image();
    auto const& mean_adc_v = mean_adc();
    bool use_mean_image = !(mean_image_v.empty());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(_adc_gaus_mean,_adc_gaus_sigma);

    const bool apply_smearing = _adc_gaus_sigma > 0.;

    for(size_t ch=0;ch<_num_channels;++ch) {

        size_t input_ch = _slice_v[ch];

        float mult_factor = 1.;
	if(apply_smearing)
	  mult_factor = (float)(d(gen));

        auto& input_img = image_v[input_ch].as_vector();
        auto const& min_adc = _min_adc_v[ch];
        auto const& max_adc = _max_adc_v[ch];
        size_t caffe_idx=0;
        size_t output_idx = ch * _rows * _cols;
        float val=0;
        if(use_mean_image) {
          auto const& mean_img = mean_image_v[input_ch].as_vector();
          for(size_t col=0; col<_cols; ++col) {
            for(size_t row=0; row<_rows; ++row) {
                auto const& input_idx = _caffe_idx_to_img_idx[caffe_idx];
		val = input_img[input_idx];
		if(apply_smearing) val *= (_adc_gaus_pixelwise ? d(gen) : mult_factor);
                val -= mean_img[input_idx];
		if( val < min_adc ) val = 0.;
		if( val > max_adc ) val = max_adc;
                _entry_data[output_idx] = val;
                ++output_idx;
                ++caffe_idx;
            }
          }
        }else{
          auto const& mean_adc = mean_adc_v[ch];
          for(size_t col=0; col<_cols; ++col) {
            for(size_t row=0; row<_rows; ++row) {
	      auto const& input_idx = _caffe_idx_to_img_idx[caffe_idx];
	      val = input_img[input_idx];
	      if(apply_smearing) val *= (_adc_gaus_pixelwise ? d(gen) : mult_factor);
	      val -= mean_adc;
	      if( val < min_adc ) val = 0.;
	      if( val > max_adc ) val = max_adc;
	      _entry_data[output_idx] = val;
	      ++output_idx;
	      ++caffe_idx;
            }
          }          
        }
    }

    // labels
    ROIType_t roi_type = kROICosmic;
    for(auto const& roi : roi_v) {
      if(roi.MCSTIndex() != kINVALID_USHORT) continue;
      roi_type = roi.Type();
      if(roi_type == kROIUnknown) roi_type = PDG2ROIType(roi.PdgCode());
      LARCV_INFO() << roi.dump() << std::endl;
      break;
    }

    // Convert type to class
    size_t caffe_class = _roitype_to_class[roi_type];

    if(caffe_class == kINVALID_SIZE) {
      LARCV_CRITICAL() << "ROIType_t " << roi_type << " is not among those defined for final set of class!" << std::endl;
      throw larbys();
    }

    _label = (float)(_roitype_to_class[roi_type]);
  }
   
}
#endif
