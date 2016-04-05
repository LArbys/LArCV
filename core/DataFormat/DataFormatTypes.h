#ifndef __DATAFORMATTYPES_H__
#define __DATAFORMATTYPES_H__

#include "Base/LArCVTypes.h"
#include <vector>
#include <set>
namespace larcv {

  static const unsigned short kINVALID_INDEX = kINVALID_USHORT;
  
  typedef unsigned short ImageIndex_t;
  typedef unsigned short ROIIndex_t;

  /// "ID" of MCShower/MCTrack in terms of its index number in the collection std::vector
  typedef unsigned short MCSTIndex_t;
  /// "ID" of MCTruth in terms of its index number in the collection std::vector
  typedef unsigned short MCTIndex_t;

  typedef unsigned short PlaneID_t;
  static const PlaneID_t kINVALID_PLANE = kINVALID_USHORT;
  
  enum ShapeType_t {
    kMCShower,
    kMCTrack,
    kUnknownShape
  };

  enum ROIType_t {
    kEminus,
    kKminus,
    kProton,
    kMuminus,
    kPiminus,
    kGamma,
    kPizero,
    kBNB,
    kCosmic,
    kUnknownROI
  };

  /// "ID" of MCParticles in terms of its G4 track ID (unless mixing multiple MC samples)
  typedef size_t MCTrackID_t;
  /// A collection of MCTrackID_t for multiple MCParticles
  typedef std::set<larcv::MCTrackID_t> MCTrackIDSet_t;
  
  enum ProductType_t {
    kProductImage2D,
    kProductROI,
    kProductUnknown
  };

}
#endif
