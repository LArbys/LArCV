#ifndef __DATAFORMATTYPES_H__
#define __DATAFORMATTYPES_H__

#include "Base/LArCVTypes.h"
#include <vector>
#include <set>
namespace larcv {

  static const unsigned short kINVALID_INDEX = kINVALID_USHORT;
  
  typedef unsigned short ImageIndex_t;
  typedef unsigned short ParticleROIIndex_t;

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

  enum ParticleType_t {
    kEminus,
    kKminus,
    kProton,
    kMuminus,
    kPiminus,
    kGamma,
    kPizero,
    kBNB,
    kCosmic,
    kUnknownParticle
  };

  /// "ID" of MCParticles in terms of its G4 track ID (unless mixing multiple MC samples)
  typedef size_t MCTrackID_t;
  /// A collection of MCTrackID_t for multiple MCParticles
  typedef std::set<larcv::MCTrackID_t> MCTrackIDSet_t;
  
  /// A collection of MCIndex_t for multiple MCShowers/MCTracks
  //typedef std::vector<larcv::MCSTIndex_t> MCIndexSet_t;
  
  /// A struct for an interaction information
  /*
  struct MCSet {
  public:
    InteractionID _id;         ///< A unique ID for an interaction
    MCPartIDSet_t _part_v;     ///< A set of particles' TrackID that belongs to the set
    MCIndexSet_t  _mcshower_v; ///< A set of particles' MCShower index number
    MCIndexSet_t  _mctrack_v;  ///< A set of particles' MCTrack index number
    MCSet(){}
    MCSet(const InteractionID& id) : _id(id) {}
    inline bool operator< (const MCSet& rhs) const { return _id < rhs._id; }
    inline bool operator< (const InteractionID& rhs) const{ return _id < rhs; }
  };
  */
  /// A set of MCSet
  //typedef std::vector<larcv::MCSet> MCSetArray;
  
}
#endif
