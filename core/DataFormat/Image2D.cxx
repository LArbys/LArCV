#include "Base/larbys.h"
#include "Image2D.h"
#include <iostream>
#include <string.h>
#include <stdio.h>
namespace larcv {

  Image2D::Image2D(size_t row_count, size_t col_count)
    : _img(row_count*col_count,0.)
    , _meta(col_count,row_count,row_count,col_count,0.,0.)
  {}
  //{import_array();}

  Image2D::Image2D(const ImageMeta& meta)
    : _img(meta.rows()*meta.cols(),0.)
    , _meta(meta)
  {}
  //{import_array();}

  Image2D::Image2D(const ImageMeta& meta, const std::vector<float>& img)
    : _img(img)
    , _meta(meta)
  { if(img.size() != meta.rows() * meta.cols()) throw larbys("Inconsistent dimensions!"); }

  Image2D::Image2D(const Image2D& rhs) 
    : _img(rhs._img)
    , _meta(rhs._meta)
  {}
  //{import_array();}
      
  Image2D::Image2D(const std::string image_file)
    : _img(0,0.)
    , _meta(1.,1.,1,1,0.,0.)
  { imread(image_file); }
  //{ imread(image_file); import_array();}

  Image2D::Image2D(ImageMeta&& meta, std::vector<float>&& img)
    : _img(std::move(img))
    , _meta(std::move(meta))
  { if(_img.size() != _meta.rows() * _meta.cols()) throw larbys("Inconsistent dimensions!"); }

  void Image2D::imread(const std::string file_name)
  {
    ::cv::Mat image;
    image = ::cv::imread(file_name.c_str(), CV_LOAD_IMAGE_COLOR);

    _img.resize(image.cols * image.rows);

    _meta = ImageMeta(image.cols,image.rows,image.cols, image.rows, 0., 0.);
      
    unsigned char* px_ptr = (unsigned char*)image.data;
    int cn = image.channels();
    
    for(int i=0;i<image.rows;i++) {
      for (int j=0;j<image.cols;j++) {
	float q = 0;
	q += (float)(px_ptr[i*image.cols*cn + j*cn + 0]);               //B
	q += (float)(px_ptr[i*image.cols*cn + j*cn + 1]) * 256.;        //G
	q += (float)(px_ptr[i*image.cols*cn + j*cn + 2]) * 256. * 256.; //R
	set_pixel(j,i,q);
      }
    }
  }

  void Image2D::resize(size_t rows, size_t cols)
  {
    _img.resize(rows * cols);
    _meta.update(rows,cols);
  }
  
  void Image2D::clear() {
    _img.clear();
    _img.resize(1,0);
    _meta.update(1,1);
    //std::cout << "[Image2D (" << this << ")] Cleared image memory " << std::endl;
  }

  void Image2D::clear_data() { for(auto& v : _img) v = 0.; }

  void Image2D::set_pixel( size_t row, size_t col, float value ) {
    if ( row >= _meta.rows() || col >= _meta.cols() )
      throw larbys("Out-of-bound pixel set request!");
    _img[ _meta.index(row,col) ] = value;
  }

  void Image2D::paint(float value)
  { for(auto& v : _img) v=value; }
  
  float Image2D::pixel( size_t row, size_t col ) const 
  { return _img[_meta.index(row,col)]; }

  void Image2D::copy(size_t row, size_t col, const float* src, size_t num_pixel) 
  { 
    const size_t idx = _meta.index(row,col);
    if(idx+num_pixel-1 >= _img.size()) throw larbys("memcpy size exceeds allocated memory!");
    memcpy(&(_img[idx]),src, num_pixel * sizeof(float));
  }

  void Image2D::copy(size_t row, size_t col, const std::vector<float>& src, size_t num_pixel) 
  {
    if(!num_pixel)
      this->copy(row,col,(float*)(&(src[0])),src.size());
    else if(num_pixel < src.size()) 
      this->copy(row,col,(float*)&src[0],num_pixel);
    else
      throw larbys("Not enough pixel in source!");
  }

  void Image2D::copy(size_t row, size_t col, const short* src, size_t num_pixel) 
  {
    const size_t idx = _meta.index(row,col);
    if(idx+num_pixel-1 >= _img.size()) throw larbys("memcpy size exceeds allocated memory!");
    for(size_t i=0; i<num_pixel; ++i) _img[idx+i] = src[num_pixel];
  }

  void Image2D::copy(size_t row, size_t col, const std::vector<short>& src, size_t num_pixel) 
  {
    if(!num_pixel)
      this->copy(row,col,(short*)(&(src[0])),src.size());
    else if(num_pixel < src.size()) 
      this->copy(row,col,(short*)&src[0],num_pixel);
    else
      throw larbys("Not enough pixel in source!");
  }

  void Image2D::reverse_copy(size_t row, size_t col, const std::vector<float>& src, size_t num_pixel)
  {
    const size_t idx = _meta.index(row,col);
    if(!num_pixel) num_pixel = src.size();
    if( (idx+1) < num_pixel ) num_pixel = idx + 1;
    for(size_t i=0; i<num_pixel; ++i) { _img[idx+i] = src[num_pixel-i-1]; }
  }

  std::vector<float> Image2D::copy_compress(size_t rows, size_t cols, CompressionModes_t mode) const
  { 
    const size_t self_cols = _meta.cols();
    const size_t self_rows = _meta.rows();
    if(self_cols % cols || self_rows % rows) {
      char oops[500];
      sprintf(oops,"Compression only possible if height/width are modular 0 of compression factor! H:%zuMOD%zu=%zu W:%zuMOD%zu=%zu",
	      self_rows,rows,self_rows%rows,self_cols,cols,self_cols%cols);
      throw larbys(oops);
    }
    size_t cols_factor = self_cols / cols;
    size_t rows_factor = self_rows / rows;
    std::vector<float> result(cols,rows);

    for(size_t col=0; col<cols; ++col) {
      for(size_t row=0; row<rows; ++row) {
	float value = 0;

	for(size_t orig_col=col*cols_factor; orig_col<(col+1)*cols_factor; ++orig_col)
	  for(size_t orig_row=row*rows_factor; orig_row<(row+1)*rows_factor; ++orig_row) {

	    if ( mode!=kMaxPool ) {
	      // for sum and average mode
	      value += _img[orig_col * self_rows + orig_row];
	    }
	    else {
	      // maxpool
	      value = ( value<_img[orig_col * self_rows + orig_row] ) ? _img[orig_col * self_rows + orig_row] : value;
	    }
	  }

	result[col*rows + row] = value;
	if ( mode==kAverage ) 
	  result[col*rows + row] /= (float)(cols_factor*cols_factor);
      }
    }
    return result;
  }

  void Image2D::compress(size_t rows, size_t cols, CompressionModes_t mode)
  { _img = copy_compress(rows,cols,mode); }

  Image2D Image2D::crop(ImageMeta& crop_meta) const
  {
    auto o_meta = _meta.overlap(crop_meta);
    size_t min_x = _meta.col(o_meta.min_x());
    size_t max_x = _meta.col(o_meta.max_x())-1;
    size_t min_y = _meta.row(o_meta.max_y());
    size_t max_y = _meta.row(o_meta.min_y())-1;
    
    ImageMeta res_meta( (max_x - min_x + 1) * _meta.pixel_width(),
			(max_y - min_y + 1) * _meta.pixel_height(),
			(max_y - min_y + 1),
			(max_x - min_x + 1),
			_meta.pos_x(min_x),
			_meta.pos_y(max_y) );
    
    std::vector<float> img;
    img.resize(res_meta.cols() * res_meta.rows());

    size_t column_size = max_y - min_y + 1;
    for(size_t x=min_x; x<=max_x; ++x)

      memcpy(&(img[(x-min_x)*column_size]), &(_img[_meta.index(min_y,x)]), column_size * sizeof(float));

    return Image2D(std::move(res_meta),std::move(img));
  }
}
