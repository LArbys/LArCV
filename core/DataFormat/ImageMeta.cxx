#ifndef __LARCAFFE_IMAGEMETA_CXX__
#define __LARCAFFE_IMAGEMETA_CXX__

#include "ImageMeta.h"
#include <sstream>
namespace larcv {

  size_t ImageMeta::index( size_t row, size_t col ) const {
    
    if ( row >= _row_count || col >= _col_count ) {
      std::stringstream ss;
      ss << "Invalid pixel index queried: (" << row << "," << col << ") but the dimension is only ("<<_row_count<<","<<_col_count<<")!"<<std::endl;
      throw larbys(ss.str());
    }    
    return ( col * _row_count + row );
  }

  long ImageMeta::col(double x, const char* callfrom_file, const int callfrom_line ) const
  {
    if(x < _origin.x || x >= (_origin.x + _width)) {
      std::stringstream ss;
      ss << "Requested col for x=" << x << " ... but the x (rows) span only " << _origin.x << " => " << _origin.x + _width << "!";
      if ( std::string(callfrom_file)=="ImageMeta.h" )
        ss << " call this using ImageMeta::col( x, __FILE__, __LINE__ ) to get info on the location of the bad call.";
      else
        ss << " calling from: " << __FILE__ << ":" << __LINE__;
      ss << std::endl;
      throw larbys(ss.str());
    }
    return (long)((x - _origin.x) / pixel_width());
  }

  long ImageMeta::row(double y, const char* callfrom_file, const int callfrom_line ) const
  {
    if(y <= (_origin.y - _height) || y > _origin.y) {
      std::stringstream ss;
      ss << "Requested row for y=" << y << " ... but the y (rows) spans only " << _origin.y - _height << " => " << _origin.y << "!";
      if ( std::string(callfrom_file)=="ImageMeta.h" )
        ss << " call this using ImageMeta::col( x, __FILE__, __LINE__ ) to get info on the location of the bad call.";
      else
        ss << " calling from: " << __FILE__ << ":" << __LINE__;
      ss << std::endl;
      throw larbys(ss.str());
    }
    return (long)((_origin.y - y) / pixel_height());
  }

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

  ImageMeta ImageMeta::inclusive(const ImageMeta& meta) const
  {
    double min_x = ( meta.min_x() < this->min_x() ? meta.min_x() : this->min_x() ); //pick smaller x min-boudn
    double max_x = ( meta.max_x() > this->max_x() ? meta.max_x() : this->max_x() ); //pick larger x max-bound

    double min_y = ( meta.min_y() < this->min_y() ? meta.min_y() : this->min_y() ); //pick smaller y min-boudn
    double max_y = ( meta.max_y() > this->max_y() ? meta.max_y() : this->max_y() ); //pick larger y max-bound

    return ImageMeta(max_x - min_x, max_y - min_y,
		     (max_y - min_y) / pixel_height(),
		     (max_x - min_x) / pixel_width(),
		     min_x, max_y, _plane);
  }

  std::string ImageMeta::dump() const
  {
    std::stringstream ss;
    ss << "Plane " << plane() << " (rows,cols) = (" << _row_count << "," << _col_count
       << ") ... Left Top (" << min_x() << "," << max_y()
       << ") ... Right Bottom (" << max_x() << "," << min_y()
       << ")" << std::endl;
    return ss.str();
  }
}

#endif
