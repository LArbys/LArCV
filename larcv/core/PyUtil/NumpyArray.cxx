#include "NumpyArray.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef USE_PYTHON3
#include "numpy/arrayobject.h"
#include <cassert>
#else
#include <numpy/ndarrayobject.h>
#endif

#include "larcv/core/Base/larcv_logger.h"

#include <iostream>
#include <sstream>


namespace larcv {

  bool NumpyArrayFloat::_setup_numpy = false;
  
  NumpyArrayFloat::NumpyArrayFloat( PyObject* array )
  {
    store(array);
  }

  int NumpyArrayFloat::store( PyObject* array )
  {
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    // cast numpy data to C-arrays
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);

    ndims = PyArray_NDIM( (PyArrayObject*)array );
    npy_intp* dims = PyArray_DIMS((PyArrayObject*)array); //new npy_intp[ndims];

    if ( ndims==1 ) {

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++)
        shape[i] = dims[i];    
      
      data.resize(dims[0]);
      for (int i=0; i<dims[0]; i++) {
        //data[i] = carray[i];
	data[i] = *((float*)PyArray_GETPTR1( (PyArrayObject*)array, i));
      }
      
    }
    else if ( ndims==2 ) {
      // float** carray;
      // int err = PyArray_AsCArray( &array, (void*)&carray, dims, ndims, descr );
      // if ( err<0 ) {
      //   std::stringstream errmsg;
      //   errmsg << "NumpyArrayFloat::store - failed to load 2D numpy array. code=" << err << std::endl;
      //   throw std::runtime_error( errmsg.str() );
      // }

      size_t totelems = 1;

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++) {
        shape[i] = dims[i];
        totelems *= dims[i];
      }
      data.resize(totelems,0);
      for (int c=0; c<dims[0]; c++) {
        for (int r=0; r<dims[1]; r++) {
          //data[ c*dims[1] + r ] =  carray[c][r];
	  data[ c*dims[1] + r ] = *((float*)PyArray_GETPTR2( (PyArrayObject*)array, c, r));
        }
      }
    }
    else if ( ndims==3 ) {

      size_t totelems = 1;

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++) {
        shape[i] = dims[i];
        totelems *= dims[i];
      }
      
      data.resize(totelems,0);
      //memcpy( data.data(), (float*)carray, sizeof(float)*totelems ); // seems like it should work
      for (int i=0; i<dims[0]; i++) {
        for (int j=0; j<dims[1]; j++) {
          for (int k=0; k<dims[2]; k++) {
            //data[ i*dims[1]*dims[2] + j*dims[2] + k ] =  carray[i][j][k];
	    data[ i*dims[1]*dims[2] + j*dims[2] + k ] = *((float*)PyArray_GETPTR3( (PyArrayObject*)array, i, j, k));
          }
        }
      }
      
    }
    else {
      throw std::runtime_error("Arrays with dimensions 4 or greater not supported");
    }

    //Py_DECREF(array);
    
    return 0;
  }

  /**
   * create a numpy array that has a copy of the data
   *
   * not entirely confident it doesn't leak memory somehow
   *
   */
  PyObject* NumpyArrayFloat::tonumpy()
  {
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    // set the shape
    npy_intp* dims = new npy_intp[ndims];
    for (int i=0; i<ndims; i++) {
      dims[i] = shape[i];
    }

    // old old way
    //PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( ndims, &dims[0], NPY_FLOAT );

    // proper way?
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT);
    PyArrayObject* array
      = (PyArrayObject*)PyArray_NewFromDescr( &PyArray_Type, descr, ndims, dims, NULL, NULL, 0, NULL);

    float* np_data = (float*)PyArray_DATA( array );
    memcpy( np_data, data.data(), sizeof(float)*data.size()); //fast, but is this causing a memory leak?
    
    delete [] dims;

    return (PyObject*)array;
  }

  /**
   * create a numpy array that wraps around the ROOT data
   *
   * can cause segfault if ROOT moves to new entry while numpy array still in use.
   * suggestion to use np.copy around this array reference
   *
   */  
  PyObject* NumpyArrayFloat::tonumpy_nocopy()
  {
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    // set the shape
    npy_intp* dims = new npy_intp[ndims];
    for (int i=0; i<ndims; i++) {
      dims[i] = shape[i];
    }

    //    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( ndims, &dims[0], NPY_FLOAT );
    //PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( ndims, &dims[0], NPY_FLOAT );
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT);
    PyArrayObject* array
      = (PyArrayObject*)PyArray_NewFromDescr( &PyArray_Type, descr, ndims, dims, NULL, data.data(), 0, NULL);

    // float* np_data = (float*)PyArray_DATA( array );
    // memcpy( np_data, data.data(), sizeof(float)*data.size()); //fast, but is this causing a memory leak?
    
    delete [] dims;

    return (PyObject*)array;
  }
  
  int NumpyArrayFloat::into_numpy2d( PyObject* pyarray )
  {
    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    if ( ndims!=2 ) {
      logger::get("PyUtil").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
				 "ERROR: this instance of NumpyArrayFloat is not 2D");
      throw std::runtime_error("ERROR: this instance of NumpyArrayFloat is not 2D");
    }
    
    // get key info to shape and types
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject*)pyarray);
    npy_intp* dims = PyArray_DIMS((PyArrayObject*)pyarray);
    int ndims = PyArray_NDIM((PyArrayObject*)pyarray);
    
    // get c-array to access data
    float **carray;
    
    if (PyArray_AsCArray(&pyarray, (void **)&carray, dims, 2, descr) < 0) {
      logger::get("PyUtil::NumpyArrayFloat").send(larcv::msg::kCRITICAL, __FUNCTION__, __LINE__,
						  "ERROR: cannot convert pyarray to 2D C-array");
      throw std::runtime_error("ERROR: cannot convert pyarray to 2D C-array");
    }

    // loop copy is slow!!
    // for (size_t i=0; i<shape[0]; i++) {
    //   for (size_t j=0; j<shape[1]; j++) {
    // 	size_t index = i*shape[1]+j;
    // 	carray[i][j] = data[index];
    //   }
    // }
    
    memcpy( (float*)carray, data.data(), sizeof(float)*data.size()); //fast, but is this causing a memory leak?
    
    
    //PyArray_Free(pyarray,  (void *)carray);
    
    return 0;
  }
  
  // ==================================================
  // NumpyArrayInt
  // ==================================================

  bool NumpyArrayInt::_setup_numpy = false;

  NumpyArrayInt::NumpyArrayInt( PyObject* array )
  {
    store(array);
  }

  int NumpyArrayInt::store( PyObject* array )
  {

    if ( !_setup_numpy ) {
      import_array1(0);
      _setup_numpy = true;
    }

    // cast numpy data to C-arrays
    const int dtype = NPY_INT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);

    ndims = PyArray_NDIM( (PyArrayObject*)array );
    npy_intp* dims = PyArray_DIMS((PyArrayObject*)array); //new npy_intp[ndims];    

    //npy_intp* dims = new npy_intp[ndims];

    if ( ndims==1 ) {

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++) {
        shape[i] = dims[i];
      }
      
      data.resize(dims[0]); 
      for (int i=0; i<dims[0]; i++) {
	data[i] = *((int*)PyArray_GETPTR1( (PyArrayObject*)array, i));
      }
    }
    else if ( ndims==2 ) {

      size_t totelems = 1;

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++) {
        shape[i] = dims[i];
        totelems *= dims[i];
      }
      data.resize(totelems,0);
      
      for (int c=0; c<dims[0]; c++) {
        for (int r=0; r<dims[1]; r++) {
          //data[ c*dims[1] + r ] =  carray[c][r];
	  data[ c*dims[1] + r ] = *((int*)PyArray_GETPTR2( (PyArrayObject*)array, c, r));
        }
      }

    }
    else if ( ndims==3 ) {

      size_t totelems = 1;

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++) {
        shape[i] = dims[i];
        totelems *= dims[i];
      }
      
      data.resize(totelems,0);
      //memcpy( data.data(), (int*)carray, sizeof(int)*totelems ); // seems like it should work
      for (int i=0; i<dims[0]; i++) {
        for (int j=0; j<dims[1]; j++) {
          for (int k=0; k<dims[2]; k++) {
            //data[ i*dims[1]*dims[2] + j*dims[2] + k ] =  carray[i][j][k];
	    data[ i*dims[1]*dims[2] + j*dims[2] + k ] = *((int*)PyArray_GETPTR3( (PyArrayObject*)array, i, j, k));	    
          }
        }
      }

    }
    else {
      throw std::runtime_error("Arrays with dimensions 4 or greater not supported");
    }

    //Py_DECREF(array);
    //delete [] dims;
    return 0;
  }

  PyObject* NumpyArrayInt::tonumpy()
  {

    import_array1(0);

    npy_intp* dims = new npy_intp[ndims];
    for (int i=0; i<ndims; i++)
      dims[i] = shape[i];

    //    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( ndims, &dims[0], NPY_INT );

    PyObject* array = PyArray_SimpleNewFromData( ndims, &dims[0], NPY_INT, data.data() );

    delete [] dims;

    return array;
  }


}
