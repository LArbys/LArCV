#ifndef __LARCV_CORE_PYUTILS_SETPYUTILS_CXX__
#define __LARCV_CORE_PYUTILS_SETPYUTILS_CXX__

#include "setpyutils.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef USE_PYTHON3
#include "numpy/arrayobject.h"
#include <cassert>
#else
#include <numpy/ndarrayobject.h>
#endif

#include "larcv/core/Base/larcv_logger.h"
#include <iostream>

namespace larcv {

  int SetPyUtil() {
    static bool once = false;
    if (!once) {
      logger::get("PyUtils").send(larcv::msg::kNORMAL, __FUNCTION__, __LINE__, "calling import_array1(0)")  << std::endl;
#ifdef USE_PYTHON3
      import_array1(0);
#else
      import_array1(0);
#endif
      once = true;
    }
    return 0;
  }
  
}


#endif
