/**
 * \file ImageMeta.h
 *
 * \ingroup LArCV
 *
 * \brief Class def header for a class ImageMeta
 *
 * @author kazuhiro
 */

/** \addtogroup LArCV

    @{*/
#ifndef __LARCV_IMAGEMETA_H__
#define __LARCV_IMAGEMETA_H__

#include <iostream>
#include <cstdio>
#include "larcv/core/Base/larbys.h"
#include "DataFormatTypes.h"
#include "Point.h"
namespace larcv {

  class Image2D;
  /**
     \class ImageMeta
     A simple class to store image's meta data which is the (tick,wire)\n
     coordinate system imposed over an image whose pixels are stored in a (row,col) matrix.
     The coordinate system properties are that:
     0) origin (left-bottom corner of the picture) absolute coordinate \n
     1) horizontal and vertical size (width and height) in double precision \n
     2) number of horizontal and vertical pixels \n
     It is meant to be associated with a larcv::Image2D object, \n
     which contains ImageMeta as an attribute.

     Note, that in previous versions, origin is the left-top corner such that
     the rows in the matrix were in tick-reverse order.
     We still can read in these tick-reversed images for backwards compatibility reasons,
     but we consider such images deprecated and do not support functions assuming
     reverse-tick order.
  */
  class ImageMeta{

    friend class Image2D;

  public:

    /// Default constructor: width, height, and origin coordinate won't be modifiable
    ImageMeta(const double width=0.,     const double height=0.,
	      const size_t row_count=0., const size_t col_count=0,
	      const double origin_x=0.,  const double origin_y=0.,
	      const PlaneID_t plane=::larcv::kINVALID_PLANE)
      : _image_id (kINVALID_INDEX)
      , _origin (origin_x,origin_y)
      , _width  (width)
      , _height (height)
      , _plane  (plane)
    {
      if( width  < 0. ) throw larbys("Width must be a positive floating point!");
      if( height < 0. ) throw larbys("Height must be a positive floating point!");
      update(row_count,col_count);
    }

    /// Default destructor
    ~ImageMeta(){}

    inline bool operator== (const ImageMeta& rhs) const
    {
      return ( _origin.x  == rhs._origin.x  &&
	       _origin.y  == rhs._origin.y  &&
	       _width     == rhs._width     &&
	       _height    == rhs._height    &&
	       _plane     == rhs._plane     &&
	       _row_count == rhs._row_count &&
	       _col_count == rhs._col_count );
    }

    inline bool operator!= (const ImageMeta& rhs) const
    { return !((*this) == rhs); }

    /// get integer index value provided for user
    ImageIndex_t image_index() const  { return _image_id; }
    /// integer provided for user
    void image_index(ImageIndex_t id) { _image_id = id;   }
    /// Top-left corner point
    const Point2D  tl   () const { return Point2D(_origin.x,          _origin.y + _height); }
    /// Bottom-left corner point
    const Point2D& bl   () const { return _origin; }
    /// Top-right corner point
    const Point2D  tr   () const { return Point2D(_origin.x + _width, _origin.y + _height); }
    /// Bottom-right corner point
    const Point2D  br   () const { return Point2D(_origin.x + _width, _origin.y          ); }
    /// PlaneID_t getter
    PlaneID_t plane     () const { return _plane;     }
    /// PlaneID_t getter
    PlaneID_t id        () const { return _plane;     }
    /// Width accessor
    double width        () const { return _width;     }
    /// Height accessor
    double height       () const { return _height;    }
    /// # rows accessor
    size_t rows         () const { return _row_count; }
    /// # columns accessor
    size_t cols         () const { return _col_count; }
    /// Pixel horizontal size
    double pixel_width  () const { return (_col_count ? _width  / (double)_col_count : 0.); }
    /// Pixel vertical size
    double pixel_height () const { return (_row_count ? _height / (double)_row_count : 0.); }

    /// Provide 1-D array index from row and column
    size_t index( size_t row, size_t col ) const;
    /// Provide absolute scale min x
    double min_x() const { return _origin.x; }
    /// Provide absolute scale max x
    double max_x() const { return _origin.x + _width; }
    /// Provide absolute scale min y
    double min_y() const { return _origin.y; }
    /// Provide absolute scale max y
    double max_y() const { return _origin.y + _height; }
    /// Provide absolute horizontal coordinate of the center of a specified pixel row
    double pos_x   (size_t col) const { return _origin.x + pixel_width()  * col; }
    /// Provide absolute vertical coordinate of the center of a specified pixel row
    double pos_y   (size_t row) const { return _origin.y + pixel_height() * row; }
    /// Provide horizontal pixel index for a given horizontal x position (in absolute coordinate)
    size_t col (double x, const char* calling_file=__FILE__, const int calling_line=__LINE__ ) const;
    
    /// Provide vertical pixel index for a given vertical y position (in absolute coordinate)
    size_t row (double y, const char* calling_file=__FILE__, const int calling_line=__LINE__ ) const;
    /// Change # of vertical/horizontal pixels in meta data
    void update(size_t row_count, size_t col_count){
      _row_count = row_count;
      _col_count = col_count;
    }
    /// Reset origin coordinate
    void   reset_origin(double x, double y) { _origin = Point2D(x,y); }
    /// Check if there's an overlap. If so return overlapping bounding box
    ImageMeta overlap(const ImageMeta& meta) const;
    /// Construct a union bounding box
    ImageMeta inclusive(const ImageMeta& meta) const;
    /// Check if (x,y) coordinate is contained in Meta
    bool contains( const float x, const float y ) const;
    /// Check if (x,y) coordinate is contained in Meta
    bool contains( const Point2D& pt ) const;

    /// produce a vector containing the xaxis coordinate values
    std::vector<float> xaxis() const;
    /// produce a vector containing the y-axis coordinate values
    std::vector<float> yaxis() const;
    
    /// Dump info in text
    std::string dump() const;

  protected:

    ImageIndex_t   _image_id; ///< Associated image ID (of the same producer name)
    larcv::Point2D _origin;   ///< Absolute coordinate of the left top corner of an image
    double    _width;         ///< Horizontal size of an image in double floating precision (in original coordinate unit size)
    double    _height;        ///< Vertical size of an image in double floating precision (in original coordinate unit size)
    size_t    _col_count;     ///< # of pixels in horizontal axis
    size_t    _row_count;     ///< # of pixels in vertical axis
    PlaneID_t _plane;         ///< unique plane ID number
  };

}

#endif
/** @} */ // end of doxygen group
