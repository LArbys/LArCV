//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// namespace
#pragma link C++ namespace larcv::chstatus+;

// enums
#pragma link C++ enum larcv::ProductType_t+;
#pragma link C++ enum larcv::ShapeType_t+;
#pragma link C++ enum larcv::ROIType_t+;
#pragma link C++ enum larcv::chstatus::ChannelStatus_t+;

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

#pragma link C++ class larcv::ChStatus+;
#pragma link C++ class std::vector<larcv::ChStatus>+;
#pragma link C++ class std::map<larcv::PlaneID_t,larcv::ChStatus>+;
#pragma link C++ class larcv::EventChStatus+;

#pragma link C++ class larcv::Pixel2D+;
#pragma link C++ class larcv::Pixel2DCluster+;
#pragma link C++ class larcv::EventPixel2D+;

#pragma link C++ class larcv::DataProductFactory+;
#pragma link C++ class larcv::IOManager+;
#ifndef __CINT__
#pragma link C++ function larcv::ProductName(larcv::ProductType_t)+;
#pragma link C++ function larcv::ROIType2String(larcv::ROIType_t)+;
#pragma link C++ function larcv::String2ROIType(const string&)+;
#endif
//ADD_NEW_CLASS ... do not change this line

#endif
