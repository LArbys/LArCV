#ifndef __SegFiller_CXX__
#define __SegFiller_CXX__

#include "SegFiller.h"
#include "DataFormat/UtilFunc.h"
#include "DataFormat/EventImage2D.h"
#include <random>

namespace larcv {

  static SegFillerProcessFactory __global_SegFillerProcessFactory__;

  SegFiller::SegFiller(const std::string name)
    : DatumFillerBase(name)
    , _slice_v()
    , _max_ch(0)
  { 
    _image_product_type  = kProductImage2D;
    _label_product_type  = kProductImage2D;
    _weight_product_type = kProductImage2D;
    _seg_channel = kINVALID_SIZE;
  }

  void SegFiller::child_configure(const PSet& cfg)
  {
    _slice_v = cfg.get<std::vector<size_t> >("Channels",_slice_v);
    _seg_channel = cfg.get<size_t>("SegChannel");
    _mirror_image = cfg.get<bool>("EnableMirror",false);
    _crop_image     = cfg.get<bool>("EnableCrop",false);
    if(_crop_image)
      _cropper.configure(cfg);
    auto type_def = cfg.get<std::vector<unsigned short> >("ClassTypeDef");
    if(type_def.size() != kROITypeMax) {
      LARCV_CRITICAL() << "ClassTypeDef length is " << type_def.size() 
		       << " but it needs to be length kROITypeMax (" << kROITypeMax << ")!" << std::endl;
      throw larbys();
    }
    for(auto const& v : type_def) {
      if(v >= kROITypeMax) {
	LARCV_CRITICAL() << "ClassTypeDef contains invalid value (" << v << ") for ROIType_t!" << std::endl;
	throw larbys();
      }
    }
    auto type_to_class = cfg.get<std::vector<unsigned short> >("ClassTypeList");
    if(type_to_class.empty()) {
      LARCV_CRITICAL() << "ClassTypeList needed to define classes!" << std::endl;
      throw larbys();
    }
    _roitype_to_class.clear();
    _roitype_to_class.resize(kROITypeMax,kINVALID_SIZE);
    _roitype_to_class[larcv::kROIUnknown] = 0;
    for(size_t i=0; i<type_to_class.size(); ++i) {
      auto const& type = type_to_class[i];
      if(type >= kROITypeMax) {
        LARCV_CRITICAL() << "ClassTypeList contains type " << type << " which is not a valid ROIType_t!" << std::endl;
        throw larbys();
      }
      _roitype_to_class[type] = i+1;
    }
    for(size_t i=0; i<_roitype_to_class.size(); ++i) {
      if(_roitype_to_class[i] != kINVALID_SIZE) continue;
      _roitype_to_class[i] = _roitype_to_class[type_def[i]];
    }

  }

  void SegFiller::child_initialize()
  { 
    _entry_image_data.clear(); 
    _entry_label_data.clear();
  }

  void SegFiller::child_batch_begin() 
  {
    _mirrored.clear();
    _mirrored.reserve(entries());
  }

  void SegFiller::child_batch_end()   
  {
    if(logger().level() <= msg::kINFO) {
      std::vector<size_t> ctr_v;
      for(auto const& v : data(kFillerLabelData)) {
        if(v>=ctr_v.size()) ctr_v.resize(v+1,0);
        ctr_v[v] += 1;
      }
      std::stringstream ss;
      ss << "Used: ";
      for(size_t i=0;i<ctr_v.size();++i)
        ss << ctr_v[i] << " of class " << i << " ... ";
      LARCV_INFO() << ss.str() << std::endl;

      size_t mirror_ctr=0;
      for(auto const& v : _mirrored) if(v) ++mirror_ctr;
        LARCV_INFO() << mirror_ctr << " / " << _mirrored.size() << " images are mirrored!" << std::endl;

    }
  }

  void SegFiller::child_finalize()    {}

  const std::vector<int> SegFiller::dim(bool image) const
  {
    std::vector<int> res(4,0);
    res[0] = entries();
    res[1] = image ? _num_channels : 1;
    res[2] = _rows;
    res[3] = _cols;
    return res;
  }

  size_t SegFiller::compute_label_size(const EventBase* label_data)
  { 
    if(entry_image_size() != kINVALID_SIZE) return entry_image_size();
    return compute_image_size(label_data);
  }

  size_t SegFiller::compute_image_size(const EventBase* image_data)
  {
    auto const& image_v = ((EventImage2D*)image_data)->Image2DArray();
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
    if ( !_crop_image ) {
      // set the dimensions from the image
      _rows = image_v.front().meta().rows();
      _cols = image_v.front().meta().cols();
    }
    else {
      // gonna crop (if speicifed dim is smaller than image dim)
      _rows = std::min( image_v.front().meta().rows(), _cropper.rows() );
      _cols = std::min( image_v.front().meta().cols(), _cropper.cols() );
    }

    // Define caffe idx to Image2D idx (assuming no crop)
    _caffe_idx_to_img_idx.resize(_rows*_cols,0);
    _mirror_caffe_idx_to_img_idx.resize(_rows*_cols,0);
    size_t caffe_idx = 0;
    for(size_t row=0; row<_rows; ++row) {
      for(size_t col=0; col<_cols; ++col) {
        _caffe_idx_to_img_idx[caffe_idx] = col*_rows + row;
        _mirror_caffe_idx_to_img_idx[caffe_idx] = (_cols-col-1)*_rows + row;
        ++caffe_idx;
      }
    }
    return (_rows * _cols * _num_channels);
  }

  void SegFiller::assert_dimension(const std::vector<larcv::Image2D>& image_v)
  {
    if(_rows == kINVALID_SIZE) {
      LARCV_WARNING() << "set_dimension() must be called prior to check_dimension()" << std::endl;
      return;
    }
    bool valid_ch   = (image_v.size() > _max_ch);
    bool valid_rows = true;
    for(auto const& img : image_v) {
      if ( !_crop_image )
        valid_rows = ( _rows == img.meta().rows() );
      if(!valid_rows) break;
    }
    bool valid_cols = true;
    for(auto const& img : image_v) {
      if ( !_crop_image )
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
      LARCV_CRITICAL() << "# of channels have changed in the input image! Image vs. MaxCh ("
      << image_v.size() << " vs. " << _max_ch << ")" << std::endl;
      throw larbys();
    }
  }

  void SegFiller::fill_entry_data( const EventBase* image_data, 
				   const EventBase* label_data,
				   const EventBase* weight_data)
  {
    LARCV_DEBUG() << "Start" << std::endl;
    auto const& image_v = ((EventImage2D*)image_data)->Image2DArray();
    this->assert_dimension(image_v);

    /* Let's not require weight to exist on all channel by default
    if(weight_data) {
      auto const& weight_v = ((EventImage2D*)weight_data)->Image2DArray();
      this->assert_dimension(weight_data);
    }
    */

    auto const& label_v = ((EventImage2D*)label_data)->Image2DArray();

    // Check label validity
    if(_seg_channel >= label_v.size()) {
      LARCV_CRITICAL() << "Segmentation image channel (" << _seg_channel << ") does not exist in data!" << std::endl;
      throw larbys();
    }

    if(_entry_image_data.empty()) _entry_image_data.resize(entry_image_size(),0.);
    for(auto& v : _entry_image_data) v = 0.;

    if(_entry_label_data.empty()) _entry_label_data.resize(entry_label_size(),0.);
    for(auto& v : _entry_label_data) v = 0.;

    if(_entry_weight_data.empty()) _entry_weight_data.resize(entry_weight_size(),0.);
    for(auto& v : _entry_weight_data) v = 0.;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> irand(0,1);
    bool mirror_image = false;
    if(_mirror_image && irand(gen)) {
      _mirrored.push_back(true);
      mirror_image = true;
      LARCV_INFO() << "Mirroring the image..." << std::endl;
    }
    else { _mirrored.push_back(false); }

    for(size_t ch=0;ch<_num_channels;++ch) {
        
        size_t input_ch = _slice_v[ch];

	LARCV_INFO() << "Filling for an image channel " << input_ch << std::endl;

        auto const& input_img2d = image_v[input_ch];
	auto const& input_meta  = input_img2d.meta();
	_entry_meta_data.push_back(input_meta);
	
        if(!ch && _crop_image) {

	  LARCV_INFO() << "First image: setting cropping region (" 
		       << _cropper.rows() << "x" << _cropper.cols() << std::endl;
          _cropper.set_crop_region(input_meta.rows(), input_meta.cols());

	}

        auto const& input_image = (_crop_image ? _cropper.crop(input_img2d) : input_img2d.as_vector());
	LARCV_INFO() << "Input image size: " << input_image.size() << " pixels" << std::endl;

        size_t caffe_idx=0;
        size_t output_idx = ch * _rows * _cols;
	LARCV_INFO() << "Start filling image from output data index " << output_idx << " for " << _rows * _cols << " pixels" << std::endl;

	if(!weight_data) {
	  for(size_t row=0; row<_rows; ++row) {
	    for(size_t col=0; col<_cols; ++col) {
	      
	      if(mirror_image)
		
		_entry_image_data[output_idx] = input_image[_mirror_caffe_idx_to_img_idx[caffe_idx]];
	      
	      else
		
		_entry_image_data[output_idx] = input_image[_caffe_idx_to_img_idx[caffe_idx]];
	      
	      ++output_idx;
	      ++caffe_idx;
	    }
	  }
	}else{

	  auto const& weight_img2d = ((EventImage2D*)(weight_data))->Image2DArray()[input_ch];

	  // Make sure dimension matches
	  if(weight_img2d.meta().cols() != input_meta.cols() ||
	     weight_img2d.meta().rows() != input_meta.rows() ) {
	    LARCV_CRITICAL() << "Channel " << input_ch << ": weight dim (col,row) = ("
			     << weight_img2d.meta().rows() << "," << weight_img2d.meta().cols() << ")"
			     << " vs. Image dim ("
			     << input_meta.rows() << "," << input_meta.cols() << ")"
			     << std::endl;
	    throw larbys();
	  }
	  
	  auto const& weight_image = (_crop_image ? _cropper.crop(weight_img2d) : weight_img2d.as_vector());

	  for(size_t row=0; row<_rows; ++row) {
	    for(size_t col=0; col<_cols; ++col) {
	      
	      if(mirror_image) {
		
		_entry_image_data  [output_idx] = input_image  [_mirror_caffe_idx_to_img_idx[caffe_idx]];
		_entry_weight_data [output_idx] = weight_image [_mirror_caffe_idx_to_img_idx[caffe_idx]];

	      }else{
		
		_entry_image_data  [output_idx] = input_image  [_caffe_idx_to_img_idx[caffe_idx]];
		_entry_weight_data [output_idx] = weight_image [_caffe_idx_to_img_idx[caffe_idx]];

	      }

	      ++output_idx;
	      ++caffe_idx;
	    }
	  }
	}
    }

    // Label
    LARCV_INFO() << "Filling for a label channel " 
		 << _seg_channel << " (input length " << label_v.size() << ")" << std::endl;
    auto const& input_lbl2d = label_v[_seg_channel];

    auto const& input_label = (_crop_image ? _cropper.crop(input_lbl2d) : input_lbl2d.as_vector());
    LARCV_INFO() << "Input image size: " << input_label.size() << " pixels" << std::endl;

    size_t caffe_idx=0;
    size_t output_idx = 0;
    float label_value = kINVALID_SIZE;
    LARCV_INFO() << "Start filling label from output data index " << output_idx << " for " << _rows * _cols << " pixels" << std::endl;

    for(size_t row=0; row<_rows; ++row) {
      for(size_t col=0; col<_cols; ++col) {

        if(mirror_image)

          label_value = input_label[_mirror_caffe_idx_to_img_idx[caffe_idx]];

        else

          label_value = input_label[_caffe_idx_to_img_idx[caffe_idx]];

	size_t class_value = _roitype_to_class[(size_t)label_value];
	//if(label_value || (size_t)(class_value))
	//std::cout<<"Label value conversion: " << label_value << " => " << class_value << std::endl;
	if(class_value == kINVALID_SIZE) {
	  LARCV_CRITICAL() << "Found invalid ROI type in the label image: " << label_value << std::endl;
	  throw larbys();
	}

	_entry_label_data[output_idx] = float(class_value);

        ++output_idx;
        ++caffe_idx;
      }
    }
    LARCV_DEBUG() << "End" << std::endl;
  }
   
}
#endif
