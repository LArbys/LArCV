#ifndef __LARCV_SPARSE_DATA_H__
#define __LARCV_SPARSE_DATA_H__

#include <vector>
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/Image2D.h"

namespace larcv {

  class SparseImage : public std::vector<float> {

  public:

    SparseImage()
      : _id(0),_nfeatures(0)
      {};
    SparseImage( const std::vector<const larcv::Image2D*>& img_v,
                 const std::vector<float>& thresholds,
                 const std::vector<int>& require_pixel=std::vector<int>() );
    SparseImage( const std::vector<larcv::Image2D>& img_v,
                 const std::vector<float>& thresholds,
                 const std::vector<int>& require_pixel=std::vector<int>() );
    SparseImage( const int nfeatures,
                 const int npoints,
                 const std::vector<float>& data,
                 const std::vector<larcv::ImageMeta>& meta_v,
                 const int index=0);
    virtual ~SparseImage() {};

    /// Return image index ID number (should be unique within larcv::EventImage2D)
    ImageIndex_t index() const { return _id; }
    /// Index setter
    void index(ImageIndex_t n) { _id = n; }

    /// Get Data
    unsigned int nfeatures() const { return _nfeatures; };
    const std::vector<float>& pixellist() const { return _pixelarray; };
    std::vector<float>& mutable_pixellist() { return _pixelarray; };    
    
    const larcv::ImageMeta& meta(int feature_index) const {
      return _meta_v.at(feature_index);
    };

    const std::vector<larcv::ImageMeta> meta_v() { return _meta_v; };

  protected:
    
    /// Add images
    void convertImages(const std::vector<const larcv::Image2D*>& img_v,
                       const std::vector<float>& thresholds,
                       const std::vector<int>& require_pixel );

    std::vector<float> _pixelarray;
    std::vector<larcv::ImageMeta>   _meta_v;
    ImageIndex_t _id;
    unsigned int _nfeatures;    
    
  };
}
#endif
