//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#ifndef __CINT__
#pragma link C++ namespace larcv;
#pragma link C++ namespace larcv::torchutil;
#pragma link C++ function larcv::torchutil::as_tensor(const larcv::Image2D&)+;
#endif
//ADD_NEW_CLASS ... do not change this line

#endif




















