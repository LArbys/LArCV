#ifndef __LARCV_TORCHUTILS_CXX__
#define __LARCV_TORCHUTILS_CXX__

#include <iostream>
#include <cstring>
#include "TorchUtils.h"

#include "larcv/core/Base/larcv_logger.h"

namespace larcv {
  namespace torchutils {
    
#ifndef __CLING__
#ifndef __CINT__
    
  
    /**
     * returns a view of the data in the image. copy-less.
     *
     * @param[in] img Image2D
     * @return torch tensor with view of the data
     */
    torch::Tensor as_tensor( const larcv::Image2D& img ) {
      const float* data = (float*)img.as_vector().data();
      return torch::from_blob( (float*)data, { (int)img.meta().cols(), (int)img.meta().rows() } );
    }
    
    /*
     * returns an image2d from a torch:tensor. need meta
     *
     */
    larcv::Image2D image2d_fromtorch( torch::Tensor& ten, 
				      const larcv::ImageMeta& meta ) {
      // check the size
      if (ten.dim()==4 ) {
	if ( ten.size(0)!=1 || ten.size(1)!=1 || ten.size(2)!=meta.rows() || ten.size(3)!=meta.cols() )
	  larcv::logger::get("torchutils").send(larcv::msg::kERROR,__FUNCTION__,__LINE__)
	    << "4-dim tensor shape mismatch. "
	    << "tensor(1,1,row,col)=(" << ten.size(0) << "," << ten.size(1) << "," << ten.size(2) << "," << ten.size(3) << ") "
	    << "vs. meta(row,col)=(" << meta.rows() << "," << meta.cols() << ")"
	    << std::endl;
      }
      else if (ten.dim()==2) {
	if ( ten.size(0)!=meta.rows() || ten.size(1)!=meta.cols() )
	  larcv::logger::get("torchutils").send(larcv::msg::kERROR,__FUNCTION__,__LINE__)
	    << "2-dim tensor shape mismatch. "
	    << "tensor(row,col)=(" << ten.size(0) << "," << ten.size(1) << ") "
	    << "vs. meta(row,col)=(" << meta.rows() << "," << meta.cols() << ")"
	    << std::endl;	
      }
      
      larcv::Image2D out( meta );
      std::vector<float> vec = out.move();
      memcpy( vec.data(), ten.reshape( {(int)meta.rows(),(int)meta.cols()} ).data_ptr(), sizeof(float)*meta.rows()*meta.cols() );
      out.move( std::move(vec) );
      
      return out;
    }
    
#endif
#endif
    
  }
}

#endif
