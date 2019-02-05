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
#pragma link C++ namespace larcv::json+;
#pragma link C++ function  larcv::json::as_json(const larcv::Image2D& )+;
#pragma link C++ function  larcv::json::as_json(const larcv::ImageMeta& )+;
#pragma link C++ function  larcv::json::as_bson(const larcv::Image2D& )+;
#pragma link C++ function  larcv::json::image2d_from_json(const nlohmann::json& )+;
#pragma link C++ function  larcv::json::imagemeta_from_json(const nlohmann::json& )+;
#pragma link C++ function  larcv::json::image2d_from_bson(const std::vector<std::uint8_t>& )+;
#pragma link C++ class     larcv::json::load_jsonutils+;
#ifndef __CINT__
#pragma link C++ function  larcv::json::as_pystring(const larcv::Image2D&)+);
#pragma link C++ function  image2d_from_pystring( PyObject* str )+;
#endif
//ADD_NEW_CLASS ... do not change this line

#endif






