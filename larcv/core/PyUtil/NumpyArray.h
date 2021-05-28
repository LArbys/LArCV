#ifndef __LARCV_CORE_DATAFORMAT_NUMPY_ARRAY_H__
#define __LARCV_CORE_DATAFORMAT_NUMPY_ARRAY_H__

#include <Python.h>
#include "bytesobject.h"
#include <vector>

namespace larcv {

  class NumpyArrayFloat  {
    
  public:
    
    NumpyArrayFloat()
      : ndims(0)
      {};

    NumpyArrayFloat( PyObject* array );
    virtual ~NumpyArrayFloat() {};

    int store( PyObject* array );
    PyObject* tonumpy();    
    int dtype();

    int ndims;
    std::vector<int> shape;
    std::vector<float> data;


  };

}

#endif
