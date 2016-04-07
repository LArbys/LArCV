/**
 * \file Image2D.h
 *
 * \ingroup LArCV
 * 
 * \brief Class def header for an image data holder
 *
 * @author tmw, kazu
 */

/** \addtogroup LArCV

    @{*/

#ifndef __LARCV_Image2D_H__
#define __LARCV_Image2D_H__

#include <vector>
#include <cstdlib>
#include "ImageMeta.h"

namespace larcv {

  /**
     \class Image2D
     Meant to be a storage class for an image into a ROOT file. Not yet implemented (don't bother reading!)
  */
  class Image2D {
  //class Image2D : public std::vector<float> {
    
  public:
    
    Image2D(size_t row_count=0, size_t col_count=0);
    Image2D(const ImageMeta&);
    Image2D(const ImageMeta&, const std::vector<float>&);
    Image2D(const Image2D&);
    
#ifndef __CINT__
#ifndef __CLING__
    Image2D(ImageMeta&&, std::vector<float>&&);
#endif
#endif
    virtual ~Image2D(){}

    void reset(const ImageMeta&);

    enum CompressionModes_t { kSum, kAverage, kMaxPool};

    ImageIndex_t index() const { return _id; }
    void index(ImageIndex_t n) { _id = n; }

    void imread(const std::string file_name);//, bool as_imshow=false);

    float pixel(size_t row, size_t col) const;
    const ImageMeta& meta() const { return _meta; }
    std::vector<float> copy_compress(size_t row_count, size_t col_count, CompressionModes_t mode=kSum) const;
    void copy(size_t row, size_t col, const float* src, size_t num_pixel);
    void copy(size_t row, size_t col, const std::vector<float>& src, size_t num_pixel=0);
    void copy(size_t row, size_t col, const short* src, size_t num_pixel);
    void copy(size_t row, size_t col, const std::vector<short>& src, size_t num_pixel=0);
    void reverse_copy(size_t row, size_t col, const std::vector<float>& src, size_t nskip=0, size_t num_pixel=0);

    Image2D crop(ImageMeta& crop_meta) const;

    const std::vector<float>& as_vector() const 
    { return _img; }

    void resize( size_t row_count, size_t col_count );
    void set_pixel( size_t row, size_t col, float value );
    void paint(float value);
    void clear_data();
    void compress(size_t row_count, size_t col_count, CompressionModes_t mode=kSum);

  private:
    std::vector<float> _img;
    ImageIndex_t _id;
    ImageMeta _meta;
    void clear();
  };

}

#endif
/** @} */ // end of doxygen group 
