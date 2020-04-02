#include "SparseImage.h"

#include <sstream>

namespace larcv {

  SparseImage::SparseImage( const std::vector<const larcv::Image2D*>& img_v,
                            const std::vector<float>& thresholds,
                            const std::vector<int>& require_pixel )
    : _pixelarray(std::vector<float>()),
      _id(0),
      _nfeatures(img_v.size())
  {
    convertImages(img_v,thresholds,require_pixel);
  }

  SparseImage::SparseImage( const std::vector<larcv::Image2D>& img_v,
                            const std::vector<float>& thresholds,
                            const std::vector<int>& require_pixel )
    : _pixelarray(std::vector<float>()),
      _id(0),
      _nfeatures(img_v.size())
  {
    std::vector<const larcv::Image2D*> pimg_v;
    for ( auto const& img : img_v )
      pimg_v.push_back( &img );
    convertImages(pimg_v,thresholds,require_pixel);
  }

  /**
   * constructor where we convert a subset of a vector
   * we provide this to avoid a copy
   *
   */
  SparseImage::SparseImage( const std::vector<larcv::Image2D>& img_v,
                            const int start_index, const int end_index,
                            const std::vector<float>& thresholds,
                            const std::vector<int>& require_pixel )
    : _pixelarray(std::vector<float>()),
      _id(0),
      _nfeatures(end_index-start_index+1)
  {
    std::vector<const larcv::Image2D*> pimg_v;
    for ( int idx=start_index; idx<=end_index; idx++ )
      pimg_v.push_back( &img_v[idx] );
    convertImages(pimg_v,thresholds,require_pixel);
  }

  // constructor for Infill version - keep dead channels
  SparseImage::SparseImage( const larcv::Image2D& img,
                            const larcv::Image2D& labels,
                            const std::vector<float>& thresholds )
    : _pixelarray(std::vector<float>()),
      _id(0),
      _nfeatures(1)
   {

     convertImages_Infill(img, labels, thresholds);
   }

  /**
   * constructor directly using values. for copying.
   *
   */
  SparseImage::SparseImage( const int nfeatures,
                            const int npoints,
                            const std::vector<float>& data,
                            const std::vector<larcv::ImageMeta>& meta_v,
                            const int index)
    : _pixelarray(data),
      _meta_v(meta_v),
      _id(index),
      _nfeatures(nfeatures)
  {
    // sanity checks
    if ( (2+nfeatures)*npoints != (int)data.size()  ) {
      std::stringstream msg;
      msg << "SparseImage::SparseImage: "
          << " given number of features and points, "
          << " the number of data elements disagrees";
      throw std::runtime_error(msg.str());
    }

    if ( _nfeatures!=meta_v.size() ) {
      throw std::runtime_error("SparseImage::SparseImage: number of features and metas must be the same");
    }
  }


  void SparseImage::convertImages( const std::vector<const larcv::Image2D*>& img_v,
                                   const std::vector<float>& thresholds,
                                   const std::vector<int>& require_pixel )
  {

    if ( img_v.size()==0 ) {
      throw std::runtime_error("SparseImage::convertImages: input image vector is empty");
    }

    // save metas
    for (auto const& pimg : img_v )
      _meta_v.push_back( pimg->meta() );

    size_t ncols  = _meta_v.front().cols();
    size_t nrows  = _meta_v.front().rows();
    size_t nfeats = nfeatures();

    std::vector<float> thresh = thresholds;
    if (thresh.size()==1) {
      // broadcast
      thresh.resize( nfeats, thresholds[0] );
    }
    else if ( thresh.size()!=img_v.size() ) {
      throw std::runtime_error("SparseImage::SparseImage: number of threshold values should be 1 or match number of images");
    }

    // reserve space, expect sparsity factor of about 10
    _pixelarray.clear();
    _pixelarray.reserve( ncols*nrows*nfeats/10 );

    for ( size_t col=0; col<ncols; col++ ) {
      for ( size_t row=0; row<nrows; row++ ) {
        std::vector<float> feats(nfeats,0);
        bool hasfeature = false;
        for ( size_t ifeat=0; ifeat<img_v.size(); ifeat++ ) {
          float pixval = img_v[ifeat]->pixel(row,col);
          feats[ifeat] = pixval;
          if ( pixval>=thresh[ifeat] && (require_pixel.size()==0 || require_pixel[ifeat]==1) ) {
            hasfeature = true;
          }
        }

        if ( hasfeature ) {
          _pixelarray.push_back((float)row);
          _pixelarray.push_back((float)col);
          for ( auto& val : feats )
            _pixelarray.push_back( val );
        }
      }
    }
  }

// convert Images version built for infill
  void SparseImage::convertImages_Infill( const larcv::Image2D& img,
                           const larcv::Image2D& labels,
                           const std::vector<float>& thresholds )
  {

    // save meta
    _meta_v.push_back(img.meta());

    size_t ncols  = _meta_v.front().cols();
    size_t nrows  = _meta_v.front().rows();

    std::vector<float> thresh = thresholds;
    if (thresh.size()!=1) {
      // broadcast
      thresh.resize( 1, thresholds[0] );
    }

    // reserve space, expect sparsity factor of about 10
    _pixelarray.clear();
    _pixelarray.reserve( ncols*nrows );

    for ( size_t col=0; col<ncols; col++ ) {
      for ( size_t row=0; row<nrows; row++ ) {
        bool hasfeature = false;
        float pixval = img.pixel(row,col);
        float labelval = labels.pixel(row,col);
        if ( pixval>=thresh[0] || labelval>0.0 ) {
          hasfeature = true;
        }
        if ( hasfeature ) {
          _pixelarray.push_back((float)row);
          _pixelarray.push_back((float)col);
          _pixelarray.push_back( pixval );
        }
      }
    }

  }

  /**
  * embed data into dense image2d format
  *
  */
  std::vector<larcv::Image2D> SparseImage::as_Image2D() const {
    std::vector<larcv::Image2D> img_v;
    for (size_t iimg=0; iimg<_meta_v.size(); iimg++ ) {
      const larcv::ImageMeta& meta = _meta_v.at(iimg);
      larcv::Image2D img(meta);
      img.paint(0.0);
      img_v.emplace_back( std::move(img) );
    }

    size_t stride = 2+_nfeatures;
    size_t npts = _pixelarray.size()/stride;
    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      int row = (int)_pixelarray[ ipt*stride ];
      int col = (int)_pixelarray[ ipt*stride+1 ];

      for (size_t iimg=0; iimg<_meta_v.size(); iimg++ )
        img_v[iimg].set_pixel( row, col, _pixelarray[ ipt*stride+2+iimg ] );
    }

    return img_v;
  }
}
