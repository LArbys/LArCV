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

  Image2D::Image2D(const ImageMeta& meta)
    : _img(meta.rows()*meta.cols(),0.)
    , _meta(meta)
  {}

  Image2D::Image2D(const ImageMeta& meta, const std::vector<float>& img)
    : _img(img)
    , _meta(meta)
  { if(img.size() != meta.rows() * meta.cols()) throw larbys("Inconsistent dimensions!"); }

  Image2D::Image2D(const Image2D& rhs) 
    : _img(rhs._img)
    , _meta(rhs._meta)
  {}
      
  Image2D::Image2D(ImageMeta&& meta, std::vector<float>&& img)
    : _img(std::move(img))
    , _meta(std::move(meta))
  { if(_img.size() != _meta.rows() * _meta.cols()) throw larbys("Inconsistent dimensions!"); }

  void Image2D::reset(const ImageMeta& meta)
  {
    _meta = meta;
    if(_img.size() != _meta.rows() * _meta.cols()) _img.resize(_meta.rows() * _meta.cols());
    paint(0.);
  }

  void Image2D::resize(size_t rows, size_t cols, float fillval)
  {
    auto const current_rows = _meta.rows();
    auto const current_cols = _meta.cols();
    std::vector<float> img(rows*cols,fillval);

    size_t npixels = std::min(current_rows,rows);
    for(size_t c=0; c < std::min(cols,current_cols); ++c) {
      for(size_t r=0; r<npixels; ++r) {
	img[c*rows+r] = _img[c*current_rows+r];
      }
    }

    _meta = ImageMeta(cols * _meta.pixel_width(), rows * _meta.pixel_height(),
		      rows, cols,
		      _meta.min_x(), _meta.max_y(),
		      _meta.plane());
    _img = std::move(img);
  }
  
  void Image2D::clear() {
    _img.clear();
    _img.resize(1,0);
    _meta.update(1,1);
    //std::cout << "[Image2D (" << this << ")] Cleared image memory " << std::endl;
  }

  void Image2D::clear_data() { for(auto& v : _img) v = 0.; }

  void Image2D::set_pixel( size_t index, float value ) {
    if ( index >= _img.size() ) throw larbys("Out-of-bound pixel set request!");
    _img[ index ] = value;
  }

  void Image2D::set_pixel( size_t row, size_t col, float value ) {
    if ( row >= _meta.rows() || col >= _meta.cols() )
      throw larbys("Out-of-bound pixel set request!");
    _img[ _meta.index(row,col) ] = value;
  }

  void Image2D::paint(float value)
  { for(auto& v : _img) v=value; }

  void Image2D::paint_row(int row, float value)
  { 
    for ( size_t col=0; col<_meta.cols(); col++ )
      set_pixel( row, col, value );
  }

  void Image2D::paint_col(int col, float value)
  { 
    for ( size_t row=0; row<_meta.rows(); row++ )
      set_pixel( row, col, value );
  }

  void Image2D::threshold(float thres, float lower_overwrite)
  { for(auto& v : _img) if( v <= thres ) v = lower_overwrite; }

  void Image2D::binary_threshold(float thres, float lower_overwrite, float upper_overwrite)
  { for(auto& v : _img) v = (v <= thres ? lower_overwrite : upper_overwrite); }

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
    else if(num_pixel <= src.size()) 
      this->copy(row,col,(float*)&src[0],num_pixel);
    else
      throw larbys("Not enough pixel in source!");
  }

  void Image2D::copy(size_t row, size_t col, const short* src, size_t num_pixel) 
  {
    const size_t idx = _meta.index(row,col);
    if(idx+num_pixel-1 >= _img.size()) throw larbys("memcpy size exceeds allocated memory!");
    for(size_t i=0; i<num_pixel; ++i) _img[idx+i] = src[idx];
  }

  void Image2D::copy(size_t row, size_t col, const std::vector<short>& src, size_t num_pixel) 
  {
    if(!num_pixel)
      this->copy(row,col,(short*)(&(src[0])),src.size());
    else if(num_pixel <= src.size()) 
      this->copy(row,col,(short*)&src[0],num_pixel);
    else
      throw larbys("Not enough pixel in source!");
  }

  void Image2D::reverse_copy(size_t row, size_t col, const std::vector<float>& src, size_t nskip, size_t num_pixel)
  {
    size_t idx = 0;
    try{
      idx = _meta.index(row,col);
    }catch(const larbys& err){
      std::cout << "Exception caught @ " << __FUNCTION__ << std::endl
		<< "Image2D ... fill row: "<<row<<" => "<<(row+num_pixel-1)<<std::endl
		<< "Image2D ... orig idx: "<<nskip<<" => "<<nskip+num_pixel-1<<std::endl
		<< "Re-throwing exception..."<<std::endl; 
      throw err;
    }
    if(!num_pixel) num_pixel = src.size() - nskip;
    if( (idx+1) < num_pixel ) num_pixel = idx + 1;
    for(size_t i=0; i<num_pixel; ++i) { _img[idx+i] = src[nskip+num_pixel-i-1]; }
  }

  void Image2D::reverse_copy(size_t row, size_t col, const std::vector<short>& src, size_t nskip, size_t num_pixel)
  {
    size_t idx = 0;
    try{
      idx = _meta.index(row,col);
    }catch(const larbys& err){
      std::cout << "Exception caught @ " << __FUNCTION__ << std::endl
		<< "Image2D ... fill row: "<<row<<" => "<<(row+num_pixel-1)<<std::endl
		<< "Image2D ... orig idx: "<<nskip<<" => "<<nskip+num_pixel-1<<std::endl
		<< "Re-throwing exception..."<<std::endl; 
      throw err;
    }
    if(!num_pixel) num_pixel = src.size() - nskip;
    if( (idx+1) < num_pixel ) num_pixel = idx + 1;
    for(size_t i=0; i<num_pixel; ++i) { _img[idx+i] = (float)(src[nskip+num_pixel-i-1]); }
  }

  std::vector<float> Image2D::copy_compress(size_t rows, size_t cols, CompressionModes_t mode) const
  { 
    if(mode == kOverWrite) 
      throw larbys("kOverWrite is invalid for copy_compress!");

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
    std::vector<float> result(cols*rows,0);

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
	  result[col*rows + row] /= (float)(rows_factor*cols_factor);
      }
    }
    return result;
  }

  void Image2D::compress(size_t rows, size_t cols, CompressionModes_t mode)
  {
    _img = copy_compress(rows,cols,mode);
    _meta = ImageMeta(_meta.width(),_meta.height(),rows,cols,_meta.min_x(),_meta.max_y(),_meta.plane());
  }

  Image2D Image2D::crop(const ImageMeta& crop_meta) const
  {
    // Croppin region must be within the image
    if( crop_meta.min_x() < _meta.min_x() || crop_meta.min_y() < _meta.min_y() ||
	crop_meta.max_x() > _meta.max_x() || crop_meta.max_y() > _meta.max_y() )
      throw larbys("Cropping region contains region outside the image!");
    
    size_t min_col = _meta.col(crop_meta.min_x() + _meta.pixel_width()  / 2. );
    size_t max_col = _meta.col(crop_meta.max_x() - _meta.pixel_width()  / 2. );
    size_t min_row = _meta.row(crop_meta.max_y() - _meta.pixel_height() / 2. );
    size_t max_row = _meta.row(crop_meta.min_y() + _meta.pixel_height() / 2. );
    /*
    std::cout<<"Cropping! Requested:" << std::endl
	     << crop_meta.dump() << std::endl
	     <<"Original:"<<std::endl
	     <<_meta.dump()<<std::endl;
    
    std::cout<<min_col<< " => " << max_col << " ... " << min_row << " => " << max_row << std::endl;
    std::cout<<_meta.width() << " / " << _meta.cols() << " = " << _meta.pixel_width() << std::endl;
    */
    ImageMeta res_meta( (max_col - min_col + 1) * _meta.pixel_width(),
			(max_row - min_row + 1) * _meta.pixel_height(),
			(max_row - min_row + 1),
			(max_col - min_col + 1),
			_meta.min_x() + min_col * _meta.pixel_width(),
			_meta.max_y() - min_row * _meta.pixel_height(),
			_meta.plane());
    
    std::vector<float> img;
    img.resize(res_meta.cols() * res_meta.rows());

    size_t column_size = max_row - min_row + 1;
    for(size_t col=min_col; col<=max_col; ++col)

      memcpy(&(img[(col-min_col)*column_size]), &(_img[_meta.index(min_row,col)]), column_size * sizeof(float));

    //std::cout<<"Cropped:" << std::endl << res_meta.dump()<<std::endl;
    
    return Image2D(std::move(res_meta),std::move(img));
  }


  void Image2D::overlay(const Image2D& rhs, CompressionModes_t mode )
  {
    auto const& rhs_meta = rhs.meta();

    if(rhs_meta.pixel_height() != _meta.pixel_height() || rhs_meta.pixel_width() != _meta.pixel_width())

      throw larbys("Overlay not supported yet for images w/ different pixel size!");

    double x_min = std::max(_meta.min_x(),rhs_meta.min_x());
    double x_max = std::min(_meta.max_x(),rhs_meta.max_x());
    if(x_min >= x_max) return;

    double y_min = std::max(_meta.min_y(),rhs_meta.min_y());
    double y_max = std::min(_meta.max_y(),rhs_meta.max_y());
    if(y_min >= y_max) return;

    size_t row_min1 = _meta.row(y_max);
    size_t col_min1 = _meta.col(x_min);

    size_t row_min2 = rhs_meta.row(y_max);
    size_t col_min2 = rhs_meta.col(x_min);

    size_t nrows = (y_max - y_min) / _meta.pixel_height();
    size_t ncols = (x_max - x_min) / _meta.pixel_width();

    auto const& img2 = rhs.as_vector();
    
    for(size_t col_index=0; col_index < ncols; ++col_index) {

      size_t index1 = _meta.index(row_min1,col_min1+col_index);
      size_t index2 = rhs_meta.index(row_min2,col_min2+col_index);

      switch(mode) {

      case kSum:

	for(size_t row_index=0; row_index < nrows; ++row_index) 

	  _img[index1+row_index] += img2[index2+row_index];

	break;

      case kAverage:

	for(size_t row_index=0; row_index < nrows; ++row_index) 

	  _img[index1+row_index] += (_img[index1+row_index] + img2[index2+row_index]) / 2.;

	break;

      case kMaxPool:

	for(size_t row_index=0; row_index < nrows; ++row_index) 

	  _img[index1+row_index] = std::max(_img[index1+row_index],img2[index2+row_index]);

	break;
	
      case kOverWrite:

	for(size_t row_index=0; row_index < nrows; ++row_index) 

	  _img[index1+row_index] = img2[index2+row_index];

	break;
      }
    }
  }

  std::vector<float>&& Image2D::move()
  { return std::move(_img); }

  void Image2D::move(std::vector<float>&& data)
  { _img = std::move(data); }

  Image2D& Image2D::operator+=(const std::vector<float>& rhs)
  {
    if(rhs.size()!=_img.size()) throw larbys("Cannot call += uniry operator w/ incompatible size!");
    for(size_t i=0; i<_img.size(); ++i) _img[i] += rhs[i];
    return (*this);
  }

  Image2D& Image2D::operator+=(const larcv::Image2D& rhs)
  {
    if(rhs.size()!=_img.size()) throw larbys("Cannot call += uniry operator w/ incompatible size!"); 
    for (size_t col=0; col<meta().cols(); col++) {
      for ( size_t row=0; row<meta().rows(); row++ ) {
        float val = pixel(row,col);
        set_pixel(row,col,val+rhs.pixel(row,col));
      }
    }
    return (*this);
  }

  Image2D& Image2D::operator-=(const std::vector<float>& rhs)
  {
    if(rhs.size()!=_img.size()) throw larbys("Cannot call += uniry operator w/ incompatible size!");
    for(size_t i=0; i<_img.size(); ++i) _img[i] -= rhs[i];
    return (*this);
  }

  Image2D Image2D::multiRHS( const Image2D& rhs ) const {
    // check multiplication is valid
    if ( meta().cols()!=rhs.meta().rows() ) {
      char oops[500];
      sprintf( oops, "Image2D Matrix multiplication not valid. LHS cols (%zu) != RHS rows (%zu).", meta().cols(), rhs.meta().rows() );
      throw larbys(oops);
    }

    // LHS copies internal data
    Image2D out( *this );
    out.resize( (*this).meta().rows(), rhs.meta().cols() );
    for (int r=0; r<(int)(out.meta().rows()); r++) {
      for (int c=0; c<(int)(out.meta().cols()); c++) {
	float val = 0.0;
	for (int k=0; k<(int)(meta().cols()); k++) {
	  val += pixel( r, k )*rhs.pixel(k,c);
	}
	out.set_pixel( r, c, val );
      }
    }

    return out;
  }

  Image2D Image2D::operator*(const Image2D& rhs) const {
    return (*this).multiRHS( rhs );
  }

  Image2D& Image2D::operator*=( const Image2D& rhs ) {
    (*this) = multiRHS( rhs );
    return (*this);
  }

  void Image2D::eltwise(const std::vector<float>& arr,bool allow_longer) {
    // check multiplication is valid
    if ( !( (allow_longer && _img.size() <= arr.size()) || arr.size() == _img.size() ) ) {
      char oops[500];
      sprintf( oops, "Image2D element-wise multiplication not valid. LHS size = %zu (row=%zu col=%zu) while argument size = %zu",
	       _img.size(),_meta.cols(), _meta.cols(), arr.size());
      throw larbys(oops);
    }
    // Element-wise multiplication: do random access as dimension has already been checked
    for (size_t i=0; i<_img.size(); ++i)
      _img[i] *= arr[i];
  }

  void Image2D::eltwise( const Image2D& rhs ) {
    // check multiplication is valid
    auto const& meta = rhs.meta();
    if(meta.cols() != _meta.cols() || meta.rows() != _meta.rows()) {
      char oops[500];
      sprintf( oops, "Image2D element-wise multiplication not valid. LHS cols (%zu) != RHS rcols (%zu) or LHS rows(%zu)!=RHS rows(%zu)", 
	       _meta.cols(), meta.cols(), _meta.rows(), meta.rows() );
      throw larbys(oops);
    }

    eltwise(rhs.as_vector(),false);
  }
  
  void Image2D::modifyMeta( const ImageMeta& newmeta ) {
    // we do a copy and swap -- meta small enough where copy is ok
    // also, the nrows*ncols must match the underlying data. we can change the coordinates though
    if ( meta().rows()!=newmeta.rows() || meta().cols()!=newmeta.cols() ) {
      char nope[500];
      sprintf( nope, "Can only modify meta when number of rows and colums are the same. we can only change the coordinate system." );
      throw larbys(nope);
    }
    ImageMeta m(newmeta);
    std::swap(m,_meta);
  }

  void Image2D::copy_region( size_t dest_row_start, size_t dest_col_start, size_t src_row_start, size_t nrows, size_t src_col_start, size_t ncols, const larcv::Image2D& src ) {
    
    for (size_t c=0; c<ncols; c++) {
      size_t dest_index = meta().index( dest_row_start, dest_col_start+c );
      size_t src_index = src.meta().index( src_row_start, src_col_start+c );
      memcpy( &_img[dest_index], &src.as_vector()[src_index], nrows*sizeof(float) ); // much, much faster
    }
    
  }

  void Image2D::copy_region(  const larcv::Image2D& src ) {
    // we assume same pixel scale
    
    // we in bounds?
    if ( meta().min_x() > src.meta().max_x()
	 || meta().max_x() < src.meta().min_x()
	 || meta().min_y() > src.meta().max_y()
	 || meta().max_y() < src.meta().min_y() )
      return; // no overlap, so no copy
    
    //size_t dest_row_start, size_t dest_col_start, size_t src_row_start, size_t nrows, size_t src_col_start, size_t ncols;
    size_t src_col_start = 0;
    size_t des_col_start = 0;
    if ( meta().min_x() > src.meta().min_x() )
      src_col_start = src.meta().col( meta().min_x() );
    else
      des_col_start = meta().col( src.meta().min_x() );
    
    size_t src_row_start = 0;
    size_t des_row_start = 0;
    if ( meta().min_y() > src.meta().min_y() )
      src_row_start = src.meta().row( meta().min_y() );
    else
      des_row_start = meta().row( src.meta().min_y() );
    
    size_t src_col_end = src.meta().cols()-1;
    size_t des_col_end = meta().cols()-1;
    if ( meta().max_x() < src.meta().max_x() )
      src_col_end = src.meta().col( meta().max_x() );
    else
      des_col_end = meta().col( src.meta().max_x() );

    size_t src_row_end = src.meta().rows()-1;
    size_t des_row_end = meta().rows()-1;
    if ( meta().max_y() < src.meta().max_y() )
      src_row_end = src.meta().row( meta().max_y() );
    else
      des_row_end = meta().row( src.meta().max_y() );

    size_t src_nrows = src_row_end - src_row_start + 1;
    size_t src_ncols = src_col_end - src_col_start + 1;
    size_t des_nrows = des_row_end - des_row_start + 1;
    size_t des_ncols = des_col_end - des_col_start + 1;

    size_t nrows = ( src_nrows > des_nrows ) ? des_nrows : src_nrows;
    size_t ncols = ( src_ncols > des_ncols ) ? des_ncols : src_ncols;

    // std::cout << "copy_region. "
    // 	      << " src start (r,c)=(" << src_row_start << "," << src_col_start << ") "
    // 	      << " det start (r,c)=(" << des_row_start << "," << des_col_start << ") "
    // 	      << " : "
    // 	      << " src end (r,c)=(" << src_row_end << "," << src_col_end << ") "
    // 	      << " det end (r,c)=(" << des_row_end << "," << des_col_end << ") "
    // 	      << " ncols=" << ncols
    // 	      << " nrows=" << nrows
    // 	      << std::endl;
    
    // for (size_t r=0; r<nrows; r++) {
    //   for (size_t c=0; c<ncols; c++) {
    // 	set_pixel( des_row_start+r, des_col_start+c, src.pixel(src_row_start+r,src_col_start+c) );
    //   }
    // }

    // use memcpy: this is nearly 100 times faster than above loop!
    // rows are the contiguous dimension
    for (size_t c=0; c<ncols; c++) {
      size_t des_index = meta().index( des_row_start, des_col_start+c );
      size_t src_index = src.meta().index( src_row_start, src_col_start+c );
      memcpy( &_img[des_index], &src.as_vector()[src_index], nrows*sizeof(float) ); // faster?
    }
    
  }
  
}
