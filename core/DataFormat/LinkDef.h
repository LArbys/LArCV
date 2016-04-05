//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// enums
#pragma link C++ enum larcv::ProductType_t+;

// Classes
#pragma link C++ class larcv::Vertex+;
#pragma link C++ class larcv::EventBase+;

#pragma link C++ class larcv::Image2D+;
#pragma link C++ class std::vector<larcv::Image2D>+;
#pragma link C++ class larcv::EventImage2D+;


#pragma link C++ class larcv::ImageMeta+;
#pragma link C++ class std::vector<larcv::ImageMeta>+;

#pragma link C++ class larcv::ROI+;
#pragma link C++ class std::vector<larcv::ROI>+;
#pragma link C++ class larcv::EventROI+;

#pragma link C++ class larcv::DataProductFactory+;
#pragma link C++ class larcv::IOManager+;

//
// Functions
//
#pragma link C++ function larcv::as_ndarray(const larcv::Image2D&)+;
#pragma link C++ function larcv::as_mat(const larcv::Image2D&)+;

//ADD_NEW_CLASS ... do not change this line

#endif
















