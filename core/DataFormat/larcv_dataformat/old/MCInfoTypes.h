#ifndef __MCINFOTYPES_H__
#define __MCINFOTYPES_H__

#include <vector>
#include <map>
#include "InteractionID.h"
namespace larcv {

  enum ParticleType_t {
    kMCShower,
    kMCTrack,
    kUnknown
  };

  /// "ID" of MCParticles in terms of its G4 track ID (unless mixing multiple MC samples)
  typedef size_t MCTrackID_t;
  /// A collection of MCTrackID_t for multiple MCParticles
  typedef std::set<larcv::supera::MCTrackID_t> MCTrackIDSet_t;
  
  /// "ID" of MCShower/MCTrack in terms of its index number in the collection std::vector
  typedef size_t MCIndex_t;
  /// A collection of MCIndex_t for multiple MCShowers/MCTracks
  typedef std::vector<larcv::supera::MCIndex_t> MCIndexSet_t;
  
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
  //typedef std::vector<larcv::supera::MCSet> MCSetArray;
  
}

#endif
