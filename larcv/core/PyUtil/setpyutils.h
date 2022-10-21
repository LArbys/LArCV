#ifndef __LARCV_CORE_PYUTILS_SETPYUTILS_H__
#define __LARCV_CORE_PYUTILS_SETPYUTILS_H__

#ifndef __CLING__
#ifndef __CINT__
#include <Python.h>
#include "bytesobject.h"
#endif
#endif

namespace larcv {

  /// Utility function: call one-time-only numpy module initialization (you don't
  /// have to call)
  int SetPyUtil();
    
}

#endif

