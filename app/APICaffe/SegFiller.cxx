#ifndef __SEGFILLER_CXX__
#define __SEGFILLER_CXX__

#include "SegFiller.h"
#include "DataFormat/UtilFunc.h"
#include <random>

namespace larcv {

  static SegFillerProcessFactory __global_SegFillerProcessFactory__;

  SegFiller::SegFiller(const std::string name)
    : SegDatumFillerBase(name)
    , _slice_v()
    , _max_ch(0)
    , _max_adc_v()
    , _min_adc_v()
  {}

  void SegFiller::child_configure(const PSet& cfg)
  {
    _slice_v = cfg.get<std::vector<size_t> >("Channels",_slice_v);
    _max_adc_v = cfg.get<std::vector<float> >("MaxADC");
    _min_adc_v = cfg.get<std::vector<float> >("MinADC");    
    _adc_gaus_mean = cfg.get<double>("GausSmearingMean",1.0);
    _adc_gaus_sigma = cfg.get<double>("GuasSmearingSigma",-1.0);
    _adc_gaus_pixelwise = cfg.get<bool>("PixelWiseSmearing");
    _mirror_image = cfg.get<bool>("EnableMirror",false);
    _crop_image     = cfg.get<bool>("EnableCrop",false);
    if(_crop_image) {
      _randomize_crop = cfg.get<bool>("RandomizeCrop",false);
      _crop_cols      = cfg.get<int>("CroppedCols");
      _crop_rows      = cfg.get<int>("CroppedRows");
    }else{
      _crop_cols = _crop_rows = 0;
      _randomize_crop = false;
    }
    auto type_to_class = cfg.get<std::vector<unsigned short> >("ClassTypeList");
    if(type_to_class.empty()) {
      LARCV_CRITICAL() << "ClassTypeList needed to define classes!" << std::endl;
      throw larbys();
    }
    _roitype_to_class.clear();
    _roitype_to_class.resize(kROITypeMax,kINVALID_SIZE);

    for(size_t i=0; i<type_to_class.size(); ++i) {
      auto const& type = type_to_class[i];
      std::cout << "filling type " << type << "\n";
      std::cout << "i =  " << i << "\n";
      std::cout << "i+1 =  " << i+1 << "\n";
      if(type >= kROITypeMax) {
	LARCV_CRITICAL() << "ClassTypeList contains type " << type << " which is not a valid ROIType_t!" << std::endl;
	throw larbys();
      }

      _roitype_to_class[type] = i+1; // enum roi to caffe 0 based index (0==background??)

    }

  }

  void SegFiller::child_initialize()
  { _entry_data.clear(); }

  void SegFiller::child_batch_begin() 
  {
    _mirrored.clear();
    _mirrored.reserve(_nentries);
  }

  void SegFiller::child_batch_end()   
  {
    size_t mirror_ctr=0;
    for(auto const& v : _mirrored) if(v) ++mirror_ctr;
    LARCV_INFO() << mirror_ctr << " / " << _mirrored.size() << " images are mirrored!" << std::endl;
  }

  void SegFiller::child_finalize()    {}

  void SegFiller::set_dimension(const std::vector<larcv::Image2D>& image_v)
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
    if ( !_crop_image ) {
      // set the dimensions from the image
      _rows = image_v.front().meta().rows();
      _cols = image_v.front().meta().cols();
    }
    else {
      // gonna crop (if speicifed dim is smaller than image dim)
      _rows = std::min( (int)image_v.front().meta().rows(), _crop_rows );
      _cols = std::min( (int)image_v.front().meta().cols(), _crop_cols );
    }

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

  void SegFiller::fill_entry_data( const std::vector<larcv::Image2D>& image_v,
				   const std::vector<larcv::Image2D>& seg_image_v)

  {

    this->assert_dimension(image_v);
    this->assert_dimension(seg_image_v);

    const size_t batch_size = _rows * _cols * _num_channels;

    if(_entry_data.empty()) _entry_data.resize(batch_size,0.);
    if(_seg_entry_data.empty()) _seg_entry_data.resize(batch_size,0.);

    for(auto& v : _entry_data) v = 0.;
    for(auto& v : _seg_entry_data) v = 0.;

    auto const& mean_image_v = mean_image();
    auto const& mean_adc_v = mean_adc();
    bool use_mean_image = !(mean_image_v.empty());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> gaus(_adc_gaus_mean,_adc_gaus_sigma);
    std::uniform_int_distribution<> irand(0,1);
    bool mirror_image = false;
    if(_mirror_image && irand(gen)) {
      _mirrored.push_back(true);
      mirror_image = true;
    }
    else { _mirrored.push_back(false); }

    const bool apply_smearing = _adc_gaus_sigma > 0.;

    // the same cropping position is used across channels
    int row_offset = 0;
    int col_offset = 0;
    int img_rows = 0;
    int img_cols = 0;

    if ( _crop_image ) {

      int coldiff = std::max(0,(int)(image_v.front().meta().cols()-_crop_cols));
      int rowdiff = std::max(0,(int)(image_v.front().meta().rows()-_crop_rows));

      if ( _randomize_crop ) {
	if ( coldiff>0 ) {
	  std::uniform_int_distribution<> irand_col(0,coldiff);
	  col_offset = irand_col(gen);
	}

	if ( rowdiff>0 ) {
	  std::uniform_int_distribution<> irand_row(0,rowdiff);
	  row_offset = irand_row(gen);
	}
      }
      else {
	if ( coldiff>0 ) col_offset = (int)coldiff/2;
	if ( rowdiff>0 ) row_offset = (int)rowdiff/2;
      }
      //LARCV_DEBUG() << "Cropping. col offset=" << col_offset << " row offset=" << row_offset << std::endl;
      img_rows = image_v.front().meta().rows();
      img_cols = image_v.front().meta().cols();
    }
    
    
    float last_good_label = 0.0;

    for(size_t ch=0;ch<_num_channels;++ch) {

        size_t input_ch = _slice_v[ch];

        float mult_factor = 1.;
	if(apply_smearing)
	  mult_factor = (float)(gaus(gen));

        auto& input_img     = image_v[input_ch].as_vector();

        LARCV_DEBUG() << "input_ch: " << input_ch << seg_image_v.at(input_ch).as_vector().size();

	auto& seg_input_img = seg_image_v[input_ch].as_vector();

        auto const& min_adc = _min_adc_v[ch];
        auto const& max_adc = _max_adc_v[ch];

        size_t caffe_idx=0;
        size_t output_idx = ch * _rows * _cols;
        float val=0;

        if(use_mean_image) {
          auto const& mean_img = mean_image_v[input_ch].as_vector();
	  // col,row in output image coordinates
	  for(size_t row=0; row<_rows; ++row) {
	    for(size_t col=0; col<_cols; ++col) {
	      size_t input_idx = (mirror_image ? _mirror_caffe_idx_to_img_idx[caffe_idx] : _caffe_idx_to_img_idx[caffe_idx]); // passing value. bad?
	      if ( _crop_image ) {
		// the above indexing doesn't apply when cropping
		if ( !mirror_image )
		  input_idx = (col+col_offset)*img_rows + (row+row_offset);
		else
		  input_idx = (img_cols-(col+col_offset)-1)*img_rows + (row+row_offset);
	      }
	      val = input_img[input_idx];

	      if(apply_smearing) val *= (_adc_gaus_pixelwise ? gaus(gen) : mult_factor);
	      
	      val -= mean_img[input_idx];

	      if( val < min_adc ) val = 0.;
	      if( val > max_adc ) val = max_adc;
	      
	      _entry_data[output_idx] = val;

	      LARCV_DEBUG() << seg_input_img.at(input_idx) << ",";

	      _seg_entry_data[output_idx] = val > 0 ? (float) _roitype_to_class[ seg_input_img.at(input_idx) ] : 0;

	      //segmentation image doesn't have 1-1 with ADC image
	      //i see ADC but no segmented label, sigh, will have to hack I guess
	      if ( _seg_entry_data[output_idx] > 20 )
		_seg_entry_data[output_idx] = last_good_label;
	      else
		{ if ( _seg_entry_data[output_idx] > 0 ) last_good_label = _seg_entry_data[output_idx]; }

	      
	      // if ( _seg_entry_data[output_idx] > 20 ) {

	      // 	LARCV_CRITICAL() << "output_idx: " << output_idx 
	      // 			 << " _seg_entry_data: " << _seg_entry_data[output_idx] << "\n";

	      // 	LARCV_CRITICAL() << "val: " << val 
	      // 			 << " roitype_to_class: " << _roitype_to_class[ seg_input_img.at(input_idx) ] << "\n";

	      // 	LARCV_CRITICAL() << "seg_intput_img.at(input_idx): " <<  seg_input_img.at(input_idx)
	      // 			 << " input_idx: " << input_idx << "\n";

	      // 	throw larbys();
	      // }


	      ++output_idx;
	      ++caffe_idx;
            }
	    LARCV_DEBUG() << "\n";
          }
        }else{
          auto const& mean_adc = mean_adc_v[ch];
	  for(size_t row=0; row<_rows; ++row) {
	    for(size_t col=0; col<_cols; ++col) {
	      //auto const& input_idx = (mirror_image ? _mirror_caffe_idx_to_img_idx[caffe_idx] : _caffe_idx_to_img_idx[caffe_idx]);
	      size_t input_idx = (mirror_image ? _mirror_caffe_idx_to_img_idx[caffe_idx] : _caffe_idx_to_img_idx[caffe_idx]);
	      if ( _crop_image ) {
		// the above indexing doesn't apply when cropping
		if ( !mirror_image )
		  input_idx = (col+col_offset)*img_rows + (row+row_offset);
		else
		  input_idx = (img_cols-(col+col_offset)-1)*img_rows + (row+row_offset);
	      }
	      val = input_img[input_idx];
	      
	      if(apply_smearing) val *= (_adc_gaus_pixelwise ? gaus(gen) : mult_factor);

	      val -= mean_adc;

	      if( val < min_adc ) val = 0.;
	      if( val > max_adc ) val = max_adc;

	      _entry_data[output_idx] = val;

	      _seg_entry_data[output_idx] = val > 0 ? (float) _roitype_to_class[ seg_input_img[input_idx] ] : 0;
	      
	      
	      //no 1-1 mapping between ADC image and segmentation image, great, lets hack it
	      if ( _seg_entry_data[output_idx] > 20 )
		_seg_entry_data[output_idx] = last_good_label;
	      else
		{ if ( _seg_entry_data[output_idx] > 0 ) last_good_label = _seg_entry_data[output_idx]; }	      

	      //the fuck?
	      // if ( _seg_entry_data[output_idx] > 20 ) {

	      // 	LARCV_CRITICAL() << "output_idx: " << output_idx 
	      // 			 << " _seg_entry_data: " << _seg_entry_data[output_idx] << "\n";

	      // 	LARCV_CRITICAL() << "val: " << val 
	      // 			 << " roitype_to_class: " << _roitype_to_class[ seg_input_img.at(input_idx) ] << "\n";

	      // 	LARCV_CRITICAL() << "seg_intput_img.at(input_idx): " <<  seg_input_img.at(input_idx)
	      // 			 << " input_idx: " << input_idx << "\n";

	      // 	throw larbys();
	      // }

	      
	      ++output_idx;
	      ++caffe_idx;
            }
          }          
        }
    }


    // vic: enum value should just be inside _seg_entry_data already...

    // labels
    // ROIType_t roi_type = kROICosmic;
    // for(auto const& roi : roi_v) {
    //   if(roi.MCSTIndex() != kINVALID_USHORT) continue;
    //   roi_type = roi.Type();
    //   if(roi_type == kROIUnknown) roi_type = PDG2ROIType(roi.PdgCode());
    //   LARCV_INFO() << roi.dump() << std::endl;
    //   break;
    // }

    // // Convert type to class
    // size_t caffe_class = _roitype_to_class[roi_type];

    // if(caffe_class == kINVALID_SIZE) {
    //   LARCV_CRITICAL() << "ROIType_t " << roi_type << " is not among those defined for final set of class!" << std::endl;
    //   throw larbys();
    // }

    // _label = (float)(_roitype_to_class[roi_type]);

    _label = 0;

  }
   
}
#endif
