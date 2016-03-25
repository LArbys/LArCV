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
#pragma link C++ namespace larcv+;
#pragma link C++ class larcv::Range<int>+;
#pragma link C++ class larcv::Range<size_t>+;
#pragma link C++ class larcv::Range<double>+;
#pragma link C++ class larcv::UniqueRangeSet<int>+;
#pragma link C++ class larcv::UniqueRangeSet<size_t>+;
#pragma link C++ class larcv::UniqueRangeSet<double>+;

#pragma link C++ class larcv::Point2D+;
#pragma link C++ namespace larcv::msg+;
#pragma link C++ enum  larcv::msg::Level_t+;
#pragma link C++ class larcv::logger+;
#pragma link C++ class larcv::larcv_base+;
#pragma link C++ class larcv::larbys+;
//ADD_NEW_CLASS ... do not change this line

// Functions
//#pragma link C++ namespace larcv::test+;
//#pragma link C++ function larcv::test::bmark_insert(const size_t)+;
//#pragma link C++ function larcv::test::bmark_exclude(const size_t)+;
#endif




