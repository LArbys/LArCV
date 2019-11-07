#ifndef __LARCAFFE_IMAGEMETA_CXX__
#define __LARCAFFE_IMAGEMETA_CXX__

#include "ImageMeta.h"
#include <sstream>
namespace larcv {

  /**
     get the one-dim index of the data vector from the (row,col) pixel coordinates)
     
     @param[in] row row in 2D pixel coordinates
     @param[in] col column in 2D pixel coordinates
     @return int index of data vector (col-major), i.e. col*rows()+row
  */  
  size_t ImageMeta::index( size_t row, size_t col, const char* calling_file, const int calling_line ) const {
    
    if ( row >= _row_count || col >= _col_count ) {
      std::stringstream ss;
      ss << "Invalid pixel index queried: (" << row << "," << col << ") but the dimension is only ("<<_row_count<<","<<_col_count<<")!";
      if ( calling_file!=0 )
        ss << " Called from " << calling_file << ":L" << calling_line;
      ss <<std::endl;
      
      throw larbys(ss.str());
    }    
    return ( col * _row_count + row );
  }

  /** 
      get the pixel column from the x-coordinate

      note: the bounds are checked and an exception is thrown if out-of-bounds
      
      @param[in] x-coordinate
      @return size_t column coordinate
  */
  //size_t ImageMeta::col(double x) const
  size_t ImageMeta::col (double x, const char* calling_file, const int calling_line ) const
  {
    if(x < _origin.x || x >= (_origin.x + _width)) {
      std::stringstream ss;
      ss << "Requested col for x=" << x << " ... but the x-dim spans only " << _origin.x << " => " << _origin.x + _width << "!"
         << " called from " << calling_file << ":" << calling_line
         << std::endl;
      throw larbys(ss.str());
    }
    return (size_t)((x - _origin.x) / pixel_width());
  }

  /** 
      get the pixel row from the y-coordinate

      note: the bounds are checked and an exception is thrown if out-of-bounds
      
      @param[in] y-coordinate
      @return size_t row coordinate
  */
  size_t ImageMeta::row(double y, const char* calling_file, const int calling_line) const
  {
    if(y < _origin.y || y >= (_origin.y+_height) ) {
      std::stringstream ss;
      ss << "Requested row for y=" << y << " ... but the y-dim spans only " << _origin.y << " => " << _origin.y + _height << "!"
         << " called from " << calling_file << ":" << calling_line
         << std::endl;
      throw larbys(ss.str());
    }
    return (size_t)((y-_origin.y) / pixel_height());
  }
  
  /** 
      return the intersection of this and given meta
      
      note: exception thrown if no-overlap
      
      @param[in] y-coordinate
      @return size_t row coordinate
  */
  ImageMeta ImageMeta::overlap(const ImageMeta& meta) const
  {
    double minx = ( meta.min_x() < this->min_x() ? this->min_x() : meta.min_x()  ); //pick larger x min-bound
    double maxx = ( meta.max_x() < this->max_x() ? meta.max_x()  : this->max_x() ); //pick smaller x max-bound

    double miny = ( meta.min_y() < this->min_y() ? this->min_y() : meta.min_y()  ); //pick larger x min-bound
    double maxy = ( meta.max_y() < this->max_y() ? meta.max_y()  : this->max_y() ); //pick smaller x max-bound

    if(!(minx < maxx && miny < maxy)) {
      std::stringstream ss;
      ss << "No overlap found ... this X: " << this->min_x() << " => " << this->max_x() << " Y: " << this->min_y() << " => " << this->max_y()
	 << " ... the other X: " << meta.min_x() << " => " << meta.max_x() << " Y: " << meta.min_y() << " => " << meta.max_y() << std::endl;
      throw larbys(ss.str());
    }
    return ImageMeta(maxx - minx, maxy - miny,
		     (maxy - miny) / pixel_height(),
		     (maxx - minx) / pixel_width(),
		     minx, maxy, _plane);
  }

  /**
   *  smallest bounding that encloses both
   *
   *  note: uses 'this' meta for pixel height and row to calculate number of cols and rows in new meta.
   *
   *  @param[in] meta to check inclusive bounds
   *  @return ImageMeta with new bounds
   */
  ImageMeta ImageMeta::inclusive(const ImageMeta& meta) const
  {
    double min_x = ( meta.min_x() < this->min_x() ? meta.min_x() : this->min_x() ); //pick smaller x min-boudn
    double max_x = ( meta.max_x() > this->max_x() ? meta.max_x() : this->max_x() ); //pick larger x max-bound

    double min_y = ( meta.min_y() < this->min_y() ? meta.min_y() : this->min_y() ); //pick smaller y min-boudn
    double max_y = ( meta.max_y() > this->max_y() ? meta.max_y() : this->max_y() ); //pick larger y max-bound

    return ImageMeta(max_x - min_x, max_y - min_y,
		     (max_y - min_y) / pixel_height(),
		     (max_x - min_x) / pixel_width(),
		     min_x, min_y, _plane);
  }

  /**
   * check if (x,y) coordinate is contained in meta
   *
   * @param[in] x x-coordindate
   * @param[in] y y-coordindate
   * @return bool true if contained
   */
  bool ImageMeta::contains( const float x, const float y ) const {
    if ( min_x()<=x && x<max_x()
         && min_y()<=y && y<max_y() )
      return true;
    return false;
  }

  /**
   * check if Point2D (x,y) coordinate is contained in meta
   *
   * @param[in] pt Point2D
   * @return bool true if contained
   */
  bool ImageMeta::contains( const Point2D& pt ) const {
    if ( min_x()<=pt.x && pt.x<max_x()
         && min_y()<=pt.y && pt.y<max_y() )
      return true;
    return false;
  }
  
  /**
   *  dump key statistics of imagemeta
   *
   *  @return string with info
   */
  std::string ImageMeta::dump() const
  {
    std::stringstream ss;
    ss << "Plane " << plane() << " (cols,rows) = (" << cols() << "," << rows() << ")"
       << " ... pixel (width,height)=(" << pixel_width() << "," << pixel_height() << ")"
       << " ... Left Top (x,y)=(" << min_x() << "," << max_y() << ")"
       << " ... Right Bottom (x,y)=(" << max_x() << "," << min_y() << ")";
    return ss.str();
  }

  /**
   * produce a vector containing the xaxis values for each pixel column
   *
   * @return vector of floats
   */
  std::vector<float> ImageMeta::xaxis() const {
    std::vector<float> axis( cols() );
    float minx = min_x();
    float width = pixel_width();
    for ( size_t c=0; c<cols(); c++ )
      axis[c] = minx + c*width;
    return axis;
  }

  /**
   * produce a vector containing the xaxis values for each pixel column
   *
   * @return vector of floats
   */
  std::vector<float> ImageMeta::yaxis() const {
    std::vector<float> axis( rows() );
    float miny   = min_y();
    float height = pixel_height();
    for ( size_t r=0; r<rows(); r++ )
      axis[r] = miny + r*height;
    return axis;
  }
  
  
}

#endif
