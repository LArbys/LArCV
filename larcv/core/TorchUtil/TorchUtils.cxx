#ifndef __LARCV_TORCHUTILS_CXX__
#define __LARCV_TORCHUTILS_CXX__

#include <iostream>
#include "TorchUtils.h"


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

#endif
#endif
    
  }
}

#endif
