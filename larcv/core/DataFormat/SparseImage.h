#ifndef __LARCV_SPARSE_DATA_H__
#define __LARCV_SPARSE_DATA_H__

#include <vector>
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/Image2D.h"

namespace larcv {

  class SparseImage : public std::vector<float> {

  public:

    SparseImage()
      : _pixelarray(std::vector<float>()),_id(0),_nfeatures(0)
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
    SparseImage( const std::vector<larcv::Image2D>& img_v,
                 const int start_index, const int end_index,
                 const std::vector<float>& thresholds,
                 const std::vector<int>& require_pixel );
    
    // Infill version of constructor
    SparseImage( const larcv::Image2D& img,
                 const larcv::Image2D& labels,
                 const std::vector<float>& thresholds);

    virtual ~SparseImage() {};

    /// number of points
    size_t len() const { return _pixelarray.size()/(2+_nfeatures); };
    
    /// Return image index ID number (should be unique within larcv::EventImage2D)
    ImageIndex_t index() const { return _id; }
    /// Index setter
    void index(ImageIndex_t n) { _id = n; }

    /// Get Data
    int nfeatures() const { return _nfeatures; }; ///< features excludes coord points
    int stride() const { return _nfeatures+2; };
    const std::vector<float>& pixellist() const { return _pixelarray; };
    std::vector<float>& mutable_pixellist() { return _pixelarray; };

    /// Get point feature
    float getfeature( int point_index, int feature_index ) const { return _pixelarray[ point_index*stride() + feature_index ]; };

    const larcv::ImageMeta& meta(int feature_index) const {
      return _meta_v.at(feature_index);
    };

    const std::vector<larcv::ImageMeta>& meta_v() const { return _meta_v; };

    std::vector<larcv::Image2D> as_Image2D() const;

  protected:

    /// Add images
    void convertImages(const std::vector<const larcv::Image2D*>& img_v,
                       const std::vector<float>& thresholds,
                       const std::vector<int>& require_pixel );

    void convertImages_Infill(const larcv::Image2D& img,
                             const larcv::Image2D& labels,
                             const std::vector<float>& thresholds);

    std::vector<float> _pixelarray;
    std::vector<larcv::ImageMeta>   _meta_v;
    larcv::ImageMeta  _meta;
    ImageIndex_t _id;
    unsigned int _nfeatures;

  };
}
#endif
