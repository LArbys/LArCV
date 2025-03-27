//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class larcv::load_cvutil+;
#pragma link C++ function larcv::imread(const string file_name)+;
#pragma link C++ function larcv::imread_gray(const string file_name)+;
//#pragma link C++ function larcv::as_mat(const larcv::Image2D&)+;
//#pragma link C++ function larcv::as_mat_greyscale2bgr(const larcv::Image2D&,const float,const float)+;
//#pragma link C++ function larcv::draw_bb( cv::Mat&, const larcv::ImageMeta&, const larcv::ImageMeta&, const int, const int, const int, const int)+;
//ADD_NEW_CLASS ... do not change this line
#endif



















