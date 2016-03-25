#ifndef __LARCAFFE_IMAGEMETA_CXX__
#define __LARCAFFE_IMAGEMETA_CXX__

#include "ImageMeta.h"

namespace larcv {

  size_t ImageMeta::index( size_t row, size_t col ) const {
    
    if ( row >= _row_count || col >= _col_count ) throw larbys("Invalid pixel index queried");
    
    return ( col * _row_count + row );
  }

  size_t ImageMeta::col (double x) const
  {
    if(x < _origin.x || x > (_origin.x + _width)) throw larbys("Out of range x!");
    return (size_t)((x - _origin.x) / pixel_width() + 0.5);
  }

  size_t ImageMeta::row (double y) const
  {
    if(y < (_origin.y - _height) || y > _origin.y) throw larbys("Out of range y!");
    return (size_t)((_origin.y - y) / pixel_height() + 0.5);
  }

  ImageMeta ImageMeta::overlap(const ImageMeta& meta) const
  {
    double min_x = ( meta.tl().x < this->tl().x ? this->tl().x : meta.tl().x  ); //pick larger x min-bound
    double max_x = ( meta.tr().x < this->tr().x ? meta.tr().x  : this->tr().x ); //pick smaller x max-bound

    double min_y = ( meta.bl().y < this->bl().y ? this->bl().y : meta.bl().y  ); //pick larger x min-bound
    double max_y = ( meta.tl().y < this->tl().y ? meta.tl().y  : this->tl().y ); //pick smaller x max-bound

    if(!(min_x < max_x && min_y < max_y)) throw larbys("No overlap found");

    return ImageMeta(max_x - min_x, max_y - min_y,
		     (max_y - min_y) / pixel_height(),
		     (max_x - min_x) / pixel_width(),
		     min_x, max_y);
  }
  
}

#endif
