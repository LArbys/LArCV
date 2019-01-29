#ifndef __LARCV_TORCHUTILS_H__
#define __LARCV_TORCHUTILS_H__

//#ifndef __CLING__
//#ifndef __CINT__

#include <torch/torch.h>
#include "DataFormat/Image2D.h"


namespace larcv {
  namespace torchutils {

    /// larcv::Image2D to torch tensor
    torch::Tensor as_tensor(const larcv::Image2D &img);

    //Image2D as_image2d_meta(PyObject *, ImageMeta meta);

    //Image2D as_image2d(PyObject *);

    //ChStatus      as_chstatus( PyObject*, larcv::PlaneID_t planeid );
    //EventChStatus as_eventchstatus( PyObject* ); 

  }
}

//#endif
//#endif


#endif
