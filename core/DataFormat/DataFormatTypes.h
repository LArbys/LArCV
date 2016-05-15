#ifndef __DATAFORMATTYPES_H__
#define __DATAFORMATTYPES_H__

#include "Base/LArCVTypes.h"
#include <vector>
#include <set>
namespace larcv {

  /// Invalid rep for vector index
  static const unsigned short kINVALID_INDEX = kINVALID_USHORT;
  /// Image index type for Image2D within EventImage2D  
  typedef unsigned short ImageIndex_t;
  /// ROI index type for ROI within EventROI
  typedef unsigned short ROIIndex_t;

  /// "ID" of MCShower/MCTrack in terms of its index number in the collection std::vector
  typedef unsigned short MCSTIndex_t;
  /// "ID" of MCTruth in terms of its index number in the collection std::vector
  typedef unsigned short MCTIndex_t;
  /// "ID" for wire planes
  typedef unsigned short PlaneID_t;
  /// Invalid plane definition
  static const PlaneID_t kINVALID_PLANE = kINVALID_USHORT;

  /// Channel status constants
  namespace chstatus {
    static const short kNOTPRESENT = -1;        ///< Channel does not exist
    static const short kNEGATIVEPEDESTAL = -2;  ///< Channel not reco-ed due to pedestal < 0
    /// Standard channel status enum stored in the database
    enum ChannelStatus_t { 
      kDISCONNECTED=0, ///< Channel is not connected
			kDEAD=1,         ///< Dead channel
			kLOWNOISE=2,     ///< Abnormally low noise channel
			kNOISY=3,        ///< Abnormally high noise channel
			kGOOD=4,         ///< Good channel
			kUNKNOWN=5       ///< Channel w/ unverified status
    };
  }

  /// Object appearance type in LArTPC  
  enum ShapeType_t {
    kShapeShower,  ///< Shower
    kShapeTrack,   ///< Track
    kShapeUnknown  ///< LArbys
  };

  /// larcv::ROI types, used to define classification class typically
  enum ROIType_t {
    kROIUnknown=0, ///< LArbys
    kROICosmic,    ///< Cosmics
    kROIBNB,       ///< BNB
    kROIEminus,    ///< Electron
    kROIGamma,     ///< Gamma
    kROIPizero,    ///< Pi0
    kROIMuminus,   ///< Muon
    kROIKminus,    ///< Kaon
    kROIPiminus,   ///< Charged Pion
    kROIProton,    ///< Proton
    kROITypeMax    ///< enum element counter
  };

  /// "ID" of MCParticles in terms of its G4 track ID (unless mixing multiple MC samples)
  typedef size_t MCTrackID_t;
  /// A collection of MCTrackID_t for multiple MCParticles
  typedef std::set<larcv::MCTrackID_t> MCTrackIDSet_t;

  /// ProducerID_t to identify a unique data product within a process (for larcv::IOManager)
  typedef size_t ProducerID_t;
  /// Invalid ProducerID_t
  static const ProducerID_t kINVALID_PRODUCER=kINVALID_SIZE;

  /// Type of data product  
  enum ProductType_t {
    kProductImage2D,  ///< Image2D, EventImage2D
    kProductROI,      ///< ROI, EventROI
    kProductChStatus, ///< ChStatus, EventChStatus
    kProductUnknown   ///< LArbys
  };

}
#endif
