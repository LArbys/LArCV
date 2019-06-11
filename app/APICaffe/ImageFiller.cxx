#ifndef __IMAGEFILLER_CXX__
#define __IMAGEFILLER_CXX__

#include "ImageFiller.h"
#include "DataFormat/UtilFunc.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include <random>

namespace larcv {

  static ImageFillerProcessFactory __global_ImageFillerProcessFactory__;

  ImageFiller::ImageFiller(const std::string name)
    : ImageFillerBase(name)
    , _slice_v()
    , _max_ch(0)
  { 
    _image_product_type        = kProductImage2D;
  }

  void ImageFiller::child_configure(const PSet& cfg)
  {
    _slice_v          = cfg.get<std::vector<size_t> >("Channels",_slice_v);
    _mirror_image     = cfg.get<bool>("EnableMirror",false);
    _transpose_image  = cfg.get<bool>("EnableTranspose",false);
    _crop_image       = cfg.get<bool>("EnableCrop",false);
    _use_norm         = cfg.get<bool>("UseNomalizedImage",false);
    if(_crop_image)
      _cropper.configure(cfg);

  }

  void ImageFiller::child_initialize()
  { 
    _entry_image_data.clear(); 
    _entry_label_data.clear();
    _entry_multiplicity_data.clear();
  }

  void ImageFiller::child_batch_begin() 
  {
    _mirrored.clear();
    _mirrored.reserve(entries());
    _transposed.clear();
    _transposed.reserve(entries());
  }

  void ImageFiller::child_batch_end()   
  {
    if(logger().level() <= msg::kINFO) {
      LARCV_INFO() << "Total data size: " << data().size() << std::endl;

      size_t mirror_ctr=0;
      for(auto const& v : _mirrored) if(v) ++mirror_ctr;
      LARCV_INFO() << mirror_ctr << " / " << _mirrored.size() << " images are mirrored!" << std::endl;

      size_t transpose_ctr=0;
      for(auto const& v : _transposed) if(v) ++transpose_ctr;
      LARCV_INFO() << transpose_ctr << " / " << _transposed.size() << " images are transposed!" << std::endl;
	
    }
  }

  void ImageFiller::child_finalize()    {}

  const std::vector<int> ImageFiller::dim(bool image) const
  {
    std::vector<int> res;
    if(!image) {
      int max_class=0;
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

  size_t ImageFiller::compute_image_size(const EventBase* image_data)
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
    _transpose_caffe_idx_to_img_idx.resize(_rows*_cols,0);
    size_t caffe_idx = 0;
    for(size_t row=0; row<_rows; ++row) {
      for(size_t col=0; col<_cols; ++col) {
        _caffe_idx_to_img_idx[caffe_idx]           = col*_rows + row;
        _mirror_caffe_idx_to_img_idx[caffe_idx]    = (_cols-col-1)*_rows + row;
        _transpose_caffe_idx_to_img_idx[caffe_idx] = (_cols-col-1)*_rows + (_rows-row-1);
        ++caffe_idx;
      }
    }

    return (_rows * _cols * _num_channels);
  }

  void ImageFiller::assert_dimension(const std::vector<larcv::Image2D>& image_v)
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

  void ImageFiller::fill_entry_data( const EventBase* image_data)
  {
    auto const& image_v = ((EventImage2D*)image_data)->Image2DArray();
    this->assert_dimension(image_v);

    if(_entry_image_data.size() != entry_image_size())
      _entry_image_data.resize(entry_image_size(),0.);
    for(auto& v : _entry_image_data) v = 0.;

    std::random_device rd_mirror;
    std::random_device rd_transpose;
    std::mt19937 gen_mirror(rd_mirror());
    std::mt19937 gen_transpose(rd_transpose());
    std::uniform_int_distribution<> irand(0,1);
    bool mirror_image = false;
    bool transpose_image = false;
    bool use_norm = false;

    if(_use_norm) use_norm=true;

    
    if(_mirror_image && irand(gen_mirror) && !(irand(gen_transpose))) {
      _mirrored.push_back(true);
      mirror_image = true;
    }
    else { _mirrored.push_back(false); }
    
    if(_transpose_image && !(irand(gen_mirror)) && irand(gen_transpose)) {
      _transposed.push_back(true);
      transpose_image = true;
    }
    else { _transposed.push_back(false); }

    for(size_t ch=0;ch<_num_channels;++ch) {

        size_t input_ch = _slice_v[ch];

        auto const& input_img2d = image_v[input_ch];
	auto const& input_meta  = input_img2d.meta();
	_entry_meta_data.push_back(input_meta);

        if(_crop_image) _cropper.set_crop_region(input_meta.rows(), input_meta.cols());

        auto const & input_image = (_crop_image ? _cropper.crop(input_img2d) : input_img2d.as_vector());
	auto const & input_image_norm = input_img2d.norm_image_as_vector();
	
        size_t caffe_idx=0;
        size_t output_idx = ch * _rows * _cols;

        for(size_t row=0; row<_rows; ++row) {
          for(size_t col=0; col<_cols; ++col) {
            if(mirror_image)
	      _entry_image_data[output_idx] = input_image[_mirror_caffe_idx_to_img_idx[caffe_idx]];
	    else if (transpose_image)
	      _entry_image_data[output_idx] = input_image[_transpose_caffe_idx_to_img_idx[caffe_idx]];
	    else if (use_norm){
	      _entry_image_data[output_idx] = input_image_norm[_caffe_idx_to_img_idx[caffe_idx]];
	    }
	    else
              _entry_image_data[output_idx] = input_image[_caffe_idx_to_img_idx[caffe_idx]];

            ++output_idx;
            ++caffe_idx;
        }
      }
    }
    LARCV_DEBUG()<<"================"<<std::endl;
  }
  
}
#endif
