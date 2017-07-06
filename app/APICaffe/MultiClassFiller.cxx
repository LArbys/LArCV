#ifndef __MultiClassFiller_CXX__
#define __MultiClassFiller_CXX__

#include "MultiClassFiller.h"
#include "DataFormat/UtilFunc.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include <random>

namespace larcv {

  static MultiClassFillerProcessFactory __global_MultiClassFillerProcessFactory__;

  MultiClassFiller::MultiClassFiller(const std::string name)
    : DatumFillerBase(name)
    , _slice_v()
    , _max_ch(0)
  { 
    _image_product_type = kProductImage2D;
    _label_product_type = kProductROI;
  }

  void MultiClassFiller::child_configure(const PSet& cfg)
  {
    _slice_v = cfg.get<std::vector<size_t> >("Channels",_slice_v);
    _mirror_image = cfg.get<bool>("EnableMirror",false);
    _crop_image     = cfg.get<bool>("EnableCrop",false);
    if(_crop_image)
      _cropper.configure(cfg);
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

  void MultiClassFiller::child_initialize()
  { 
    _entry_image_data.clear(); 
    _entry_label_data.clear();
  }

  void MultiClassFiller::child_batch_begin() 
  {
    _mirrored.clear();
    _mirrored.reserve(entries());
  }

  void MultiClassFiller::child_batch_end()   
  {
    if(logger().level() <= msg::kINFO) {
      LARCV_INFO() << "Total data size: " << data().size() << std::endl;

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

  void MultiClassFiller::child_finalize()    {}

  const std::vector<int> MultiClassFiller::dim(bool image) const
  {
    std::vector<int> res;
    if(!image) {
      int max_class=0;
      for(auto const& v : _roitype_to_class) {
	if(v == kINVALID_SIZE) continue;
	if(max_class < v) max_class = v;
      }
      res.resize(entries(),max_class+1);
      return res;
    }

    res.resize(4);
    res[0] = entries();
    res[1] = _num_channels;
    res[2] = _rows;
    res[3] = _cols;
    return res;
  }

  size_t MultiClassFiller::compute_label_size(const EventBase* label_data)
  {
    size_t max_class=0;
    for(auto const& v : _roitype_to_class) {
      if(v == kINVALID_SIZE) continue;
      if(max_class < v) max_class = v;
    }
    return max_class + 1;
  }

  size_t MultiClassFiller::compute_image_size(const EventBase* image_data)
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

    LARCV_INFO() << "Rows = " << _rows << " ... Cols = " << _cols << std::endl;

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

  void MultiClassFiller::assert_dimension(const std::vector<larcv::Image2D>& image_v)
  {
    if(_rows == kINVALID_SIZE) {
      LARCV_WARNING() << "set_dimension() must be called prior to check_dimension()" << std::endl;
      return;
    }
    bool valid_ch   = (image_v.size() > _max_ch);
    bool valid_rows = true;
    for(size_t ch=0;ch<_num_channels;++ch) {
      size_t input_ch = _slice_v[ch];
      auto const& img = image_v[input_ch];

      if ( !_crop_image )
        valid_rows = ( _rows == img.meta().rows() );
      if(!valid_rows) {
	LARCV_ERROR() << "# of rows changed! (row,col): (" << _rows << "," << _cols << ") => (" 
		      << img.meta().rows() << "," << img.meta().cols() << ")" << std::endl;
	break;
      }
    }

    bool valid_cols = true;
    for(size_t ch=0;ch<_num_channels;++ch) {
      size_t input_ch = _slice_v[ch];
      auto const& img = image_v[input_ch];
      if ( !_crop_image )
        valid_cols = ( _cols == img.meta().cols() );
      if(!valid_cols) {
	LARCV_ERROR() << "# of cols changed! (row,col): (" << _rows << "," << _cols << ") => (" 
		      << img.meta().rows() << "," << img.meta().cols() << ")" << std::endl;
	break;
      }
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

  void MultiClassFiller::fill_entry_data( const EventBase* image_data, 
					  const EventBase* label_data,
					  const EventBase* weight_data)
  {
    if(weight_data) {
      LARCV_CRITICAL() << "MultiClassFiller cannot use weight data!" << std::endl;
      throw larbys();
    }

    auto const& image_v = ((EventImage2D*)image_data)->Image2DArray();
    this->assert_dimension(image_v);

    if(_entry_image_data.size() != entry_image_size())
      _entry_image_data.resize(entry_image_size(),0.);
    for(auto& v : _entry_image_data) v = 0.;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> irand(0,1);
    bool mirror_image = false;
    if(_mirror_image && irand(gen)) {
      _mirrored.push_back(true);
      mirror_image = true;
    }
    else { _mirrored.push_back(false); }

    for(size_t ch=0;ch<_num_channels;++ch) {

        size_t input_ch = _slice_v[ch];

        auto const& input_img2d = image_v[input_ch];
	auto const& input_meta  = input_img2d.meta();
	_entry_meta_data.push_back(input_meta);

        if(_crop_image) _cropper.set_crop_region(input_meta.rows(), input_meta.cols());

        auto const& input_image = (_crop_image ? _cropper.crop(input_img2d) : input_img2d.as_vector());

        size_t caffe_idx=0;
        size_t output_idx = ch * _rows * _cols;

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
    }

    // labels
    _entry_label_data.resize(entry_label_size());
    auto const& roi_v = ((EventROI*)label_data)->ROIArray();
    for(auto const& roi : roi_v) {
      if(roi.MCSTIndex() != kINVALID_USHORT) continue;
      ROIType_t roi_type = roi.Type();
      if(roi_type == kROIUnknown) roi_type = PDG2ROIType(roi.PdgCode());
      LARCV_INFO() << roi.dump() << std::endl;
      auto const& caffe_class = _roitype_to_class.at(roi_type);
      if(caffe_class == kINVALID_SIZE) {
	LARCV_CRITICAL() << "ROIType_t " << roi_type << " is not among those defined for final set of class!" << std::endl;
	for(size_t roi_index=0; roi_index<roi_v.size(); ++roi_index)
	  LARCV_CRITICAL() << "Dumping ROI " << roi_index << std::endl << roi_v[roi_index].dump() << std::endl;
	throw larbys();
      }
      _entry_label_data[caffe_class] = 1;
    }
  }
   
}
#endif
