#ifndef __LARCVTYPES_H__
#define __LARCVTYPES_H__

#include <string>
#include <utility>
#include <vector>
#include <limits>
#include <climits>


#include <vector>
#include <limits>
#include <climits>

/**
   \namespace larcv
   C++ namespace for developping LArTPC software interface to computer vision software (LArCV)
*/
namespace larcv {

  /**
     \struct Point2D
     Simple 2D point struct
  */
  struct Point2D {
    double x, y;
    Point2D(double xv=0, double yv=0) : x(xv), y(yv) {}
  };

  /// Used as an invalid value identifier for size_t
  static const size_t  kINVALID_SIZE   = std::numeric_limits< size_t         >::max();
  /// Used as an invalid value identifier for int
  static const int     kINVALID_INT    = std::numeric_limits< int            >::max();
  /// Used as an invalid value identifier for unsigned int
  const unsigned int   kINVALID_UINT   = std::numeric_limits< unsigned int   >::max();
  /// Used as an invalid value identifier for unsigned short
  static const short   kINVALID_SHORT  = std::numeric_limits< short          >::max();
  /// Used as an invalid value identifier for unsigned unsigned short
  const unsigned short kINVALID_USHORT = std::numeric_limits< unsigned short >::max();

  /// Namespace for larcv message related types
  namespace msg {

    enum Level_t { kDEBUG, kINFO, kNORMAL, kWARNING, kERROR, kCRITICAL, kMSG_TYPE_MAX };

    const std::string kStringPrefix[kMSG_TYPE_MAX] =
      {
	"     \033[94m[DEBUG]\033[00m ",  ///< kDEBUG message prefix
	"      \033[92m[INFO]\033[00m ",  ///< kINFO message prefix
	"    \033[95m[NORMAL]\033[00m ",  ///< kNORMAL message prefix
	"   \033[93m[WARNING]\033[00m ", ///< kWARNING message prefix
	"     \033[91m[ERROR]\033[00m ", ///< kERROR message prefix
	"  \033[5;1;33;41m[CRITICAL]\033[00m "  ///< kCRITICAL message prefix
      };
    ///< Prefix of message
  }

}
#endif
