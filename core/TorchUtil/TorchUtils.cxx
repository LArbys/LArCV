#ifndef __LARCV_TORCHUTILS_CXX__
#define __LARCV_TORCHUTILS_CXX__

#ifndef __CLING__
#ifndef __CINT__

#include <iostream>
#include "TorchUtils.h"
//#include "Base/larcv_logger.h"

namespace larcv {
namespace torchutil {
  
  /**
   * returns a view of the data in the image. copy-less.
   *
   * @param[in] img Image2D
   * @return torch tensor with view of the data
   */
  torch::Tensor as_tensor( const larcv::Image2D& img ) {
    const float* data = (float*)img.as_vector().data();
    return torch::from_blob( data, { (int)img.meta().cols(), (int)img.meta().rows() } );
  }

}
}

#endif
#endif
#endif
