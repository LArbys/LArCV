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
//#pragma link C++ function larcv::as_mat(const larcv::Image2D&)+;
#endif

#pragma link C++ class larcv::load_pycvutil+;
//ADD_NEW_CLASS ... do not change this line

#endif


















