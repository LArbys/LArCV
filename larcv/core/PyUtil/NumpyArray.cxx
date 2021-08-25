#include "NumpyArray.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifdef USE_PYTHON3
#include "numpy/arrayobject.h"
#include <cassert>
#else
#include <numpy/ndarrayobject.h>
#endif

#include <iostream>
#include <sstream>


namespace larcv {

  NumpyArrayFloat::NumpyArrayFloat( PyObject* array )
  {
    store(array);
  }

  int NumpyArrayFloat::store( PyObject* array )
  {
    import_array1(0);

    // cast numpy data to C-arrays
    const int dtype = NPY_FLOAT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);

    ndims = PyArray_NDIM( (PyArrayObject*)array );

    npy_intp* dims = new npy_intp[ndims];

    if ( ndims==1 ) {

      float* carray = nullptr;
      if ( PyArray_AsCArray( &array, (void*)&carray, dims, ndims, descr )<0 ) {
        throw std::runtime_error( "NumpyArrayFloat::store_array - failed to load 1D numpy array" );
      }

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++)
        shape[i] = dims[i];    
      
      data.resize(dims[0]);
      for (int i=0; i<dims[0]; i++) {
        data[i] = carray[i];        
      }
      
    }
    else if ( ndims==2 ) {
      float** carray;
      int err = PyArray_AsCArray( &array, (void*)&carray, dims, ndims, descr );
      if ( err<0 ) {
        std::stringstream errmsg;
        errmsg << "NumpyArrayFloat::store - failed to load 2D numpy array. code=" << err << std::endl;
        throw std::runtime_error( errmsg.str() );
      }

      size_t totelems = 1;

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++) {
        shape[i] = dims[i];
        totelems *= dims[i];
      }
      data.resize(totelems,0);
      //memcpy( data.data(), carray, sizeof(float)*totelems );
      for (int c=0; c<dims[0]; c++) {
        for (int r=0; r<dims[1]; r++) {
          data[ c*dims[1] + r ] =  carray[c][r];
        }
      }
      
    }
    else if ( ndims==3 ) {
      float*** carray = nullptr;
      if ( PyArray_AsCArray( &array, (void*)&carray, dims, ndims, descr )<0 ) {
        throw std::runtime_error( "NumpyArrayFloat::store - failed to load 3D numpy array" );
      }

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
            data[ i*dims[1]*dims[2] + j*dims[2] + k ] =  carray[i][j][k];
          }
        }
      }
      
    }
    else {
      throw std::runtime_error("Arrays with dimensions 4 or greater not supported");
    }

    delete [] dims;
  }

  PyObject* NumpyArrayFloat::tonumpy()
  {

    import_array1(0);

    npy_intp* dims = new npy_intp[ndims];
    for (int i=0; i<ndims; i++) {
      dims[i] = shape[i];
    }

    //    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( ndims, &dims[0], NPY_FLOAT );
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( ndims, &dims[0], NPY_FLOAT );

    float* np_data = (float*)PyArray_DATA( array );
    memcpy( np_data, data.data(), sizeof(float)*data.size());
    
    delete [] dims;

    return (PyObject*)array;
  }

  // ==================================================
  // NumpyArrayInt
  // ==================================================

  NumpyArrayInt::NumpyArrayInt( PyObject* array )
  {
    store(array);
  }

  int NumpyArrayInt::store( PyObject* array )
  {
    import_array1(0);

    // cast numpy data to C-arrays
    const int dtype = NPY_INT;
    PyArray_Descr *descr = PyArray_DescrFromType(dtype);

    ndims = PyArray_NDIM( (PyArrayObject*)array );

    npy_intp* dims = new npy_intp[ndims];

    if ( ndims==1 ) {

      int* carray = nullptr;
      if ( PyArray_AsCArray( &array, (void*)&carray, dims, ndims, descr )<0 ) {
        throw std::runtime_error( "NumpyArrayInt::store_array - failed to load 1D numpy array" );
      }

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++)
        shape[i] = dims[i];    
      
      data.resize(dims[0]);
      for (int i=0; i<dims[0]; i++) {
        data[i] = carray[i];        
      }
      
    }
    else if ( ndims==2 ) {
      int** carray;
      int err = PyArray_AsCArray( &array, (void*)&carray, dims, ndims, descr );
      if ( err<0 ) {
        std::stringstream errmsg;
        errmsg << "NumpyArrayInt::store - failed to load 2D numpy array. code=" << err << std::endl;
        throw std::runtime_error( errmsg.str() );
      }

      size_t totelems = 1;

      shape.resize(ndims,0);
      for (int i=0; i<ndims; i++) {
        shape[i] = dims[i];
        totelems *= dims[i];
      }
      data.resize(totelems,0);
      //memcpy( data.data(), carray, sizeof(int)*totelems );
      for (int c=0; c<dims[0]; c++) {
        for (int r=0; r<dims[1]; r++) {
          data[ c*dims[1] + r ] =  carray[c][r];
        }
      }
      
    }
    else if ( ndims==3 ) {
      int*** carray = nullptr;
      if ( PyArray_AsCArray( &array, (void*)&carray, dims, ndims, descr )<0 ) {
        throw std::runtime_error( "NumpyArrayInt::store - failed to load 3D numpy array" );
      }

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
            data[ i*dims[1]*dims[2] + j*dims[2] + k ] =  carray[i][j][k];
          }
        }
      }
      
    }
    else {
      throw std::runtime_error("Arrays with dimensions 4 or greater not supported");
    }

    delete [] dims;
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
