#ifndef __LARCV_TORCHUTILS_H__
#define __LARCV_TORCHUTILS_H__

#ifndef __CLING__
#ifndef __CINT__

#include <torch/torch.h>

#endif
#endif


#include "larcv/core/DataFormat/Image2D.h"

namespace larcv {
  namespace torchutils {

#ifndef __CLING__
#ifndef __CINT__

    /// larcv::Image2D to torch tensor
    torch::Tensor as_tensor(const larcv::Image2D &img);

#endif
#endif

  }
}



#endif
