//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//
// Functions
//
#ifndef __CINT__
#pragma link C++ function larcv::as_ndarray(const larcv::Image2D&)+;
#pragma link C++ function larcv::as_image2d(PyObject*,size_t,size_t)+;
#pragma link C++ function larcv::copy_array(PyObject*, const std::vector<float> &)+;
#pragma link C++ function larcv::fill_img_col( Image2D&, std::vector<short>&, const int, const int, const float pedestal=0.0);
#pragma link C++ function larcv::as_ndarray(const std::vector<float>&)+;
#pragma link C++ function larcv::play(PyObject*)+;
#pragma link C++ function larcv::as_ndarray(const larcv::EventChStatus&)+;
#pragma link C++ function larcv::as_ndarray(const larcv::ChStatus&)+;
#pragma link C++ function larcv::as_pybytes(const std::vector<std::uint8_t>&)+;
#pragma link C++ function larcv::as_pixelarray(const std::vector<std::uint8_t>&)+;
#pragma link C++ function larcv::as_union_pixelarray( const std::vector<const larcv::Image2D*>, const float, larcv::msg::Level_t );
#pragma link C++ function larcv::as_union_pixelarray( const larcv::Image2D&, const larcv::Image2D&, const float, larcv::msg::Level_t );
#pragma link C++ function larcv::as_union_pixelarray( const larcv::Image2D&, const larcv::Image2D&, const larcv::Image2D&, const float, larcv::msg::Level_t );
#pragma link C++ function larcv::as_sparseimg_ndarray( const larcv::SparseImage&, larcv::msg::Level_t )+;
#pragma link C++ function larcv::sparseimg_from_ndarray( PyObject*, const std::vector<larcv::ImageMeta>&, larcv::msg::Level_t );
#pragma link C++ function larcv::as_ndarray_mask_pixlist( const larcv::ClusterMask&, float, float )+;

#endif
#pragma link C++ class larcv::load_pyutil+;
#pragma link C++ class larcv::PyImageMaker+;
#pragma link C++ class larcv::NumpyArrayFloat+;

//ADD_NEW_CLASS ... do not change this line

#endif
