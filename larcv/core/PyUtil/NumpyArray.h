#ifndef __LARCV_CORE_DATAFORMAT_NUMPY_ARRAY_H__
#define __LARCV_CORE_DATAFORMAT_NUMPY_ARRAY_H__

/**
 * \file NumpyArray.h
 *
 * \ingroup PyUtil
 * 
 * \brief Classes to provide numpy array wrappers in pyroot scripts
 *
 * Because I am a bad person.
 * 
 * @author twongjirad
 */

struct _object;
typedef _object PyObject;

#ifndef __CINT__
#ifndef __CLING__
#include <Python.h>
#include "bytesobject.h"
#endif
#endif

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


  class NumpyArrayInt  {
    
  public:
    
    NumpyArrayInt()
      : ndims(0)
      {};

    NumpyArrayInt( PyObject* array );
    virtual ~NumpyArrayInt() {};

    int store( PyObject* array );
    PyObject* tonumpy();    
    int dtype();

    int ndims;
    std::vector<int> shape;
    std::vector<int> data;


  };
  
}

#endif
