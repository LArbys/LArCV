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
#pragma link C++ class larcv::ThreadDatumFiller+;
#endif
#pragma link C++ class larcv::DatumFillerBase+;
#pragma link C++ class larcv::SegFiller+;
#pragma link C++ class larcv::SimpleFiller+;
#pragma link C++ class larcv::ThreadFillerFactory+;
#pragma link C++ class larcv::RandomCropper+;
//ADD_NEW_CLASS ... do not change this line
#endif

























