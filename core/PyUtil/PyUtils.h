#ifndef __LARCV_PYUTILS_H__
#define __LARCV_PYUTILS_H__

struct _object;
typedef _object PyObject;

#ifndef __CLING__
#ifndef __CINT__
#include <Python.h>
#endif
#endif

#include "DataFormat/Image2D.h"
#include "DataFormat/ROI.h"

namespace larcv {
	/// Utility function: call one-time-only numpy module initialization (you don't have to call)
	void SetPyUtil();
        /// 
        PyObject* as_ndarray(const std::vector<float>& data);
	/// larcv::Image2D to numpy array converter
	PyObject* as_ndarray(const Image2D& img);
	/// larcv::Image2D to numpy array converter
	PyObject* as_caffe_ndarray(const Image2D& img);

        Image2D as_image2d_meta(PyObject*,ImageMeta meta);

        Image2D as_image2d(PyObject*);
}

#endif
