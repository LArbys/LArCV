//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// Classes
#pragma link C++ class larcv::ImageMeta+;
#pragma link C++ class larcv::Image2D+;
#pragma link C++ class std::vector<larcv::ImageMeta>+;
#pragma link C++ class std::vector<larcv::Image2D>+;
//#pragma link C++ class larcv::Image2DArray+;
//#pragma link C++ class larcv::IOManager<larcv::Image2DArray>+;
#pragma link C++ class larcv::IOManager<larcv::ImageMeta>+;
#pragma link C++ class larcv::IOManager<std::vector<larcv::ImageMeta> >+;
#pragma link C++ class larcv::IOManager<larcv::Image2D>+;
#pragma link C++ class larcv::IOManager<std::vector<larcv::Image2D> >+;
//ADD_NEW_CLASS ... do not change this line

#endif





