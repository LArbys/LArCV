#ifndef __SUPERA_FMWKINTERFACE_H__
#define __SUPERA_FMWKINTERFACE_H__

#include "Base/PSet.h"
#include "DataFormat/mctruth.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/simch.h"
#include "DataFormat/wire.h"
#include "DataFormat/hit.h"
#include "DataFormat/ophit.h"
#include "DataFormat/opflash.h"
#include "DataFormat/opdetwaveform.h"

//
// Data types
//
namespace supera {

  typedef ::larcv::PSet Config_t;
  
  typedef ::larlite::wire     LArWire_t;
  typedef ::larlite::opdetwaveform LArOpDigit_t;
  typedef ::larlite::hit      LArHit_t;
  typedef ::larlite::mctruth  LArMCTruth_t;
  typedef ::larlite::mctrack  LArMCTrack_t;
  typedef ::larlite::mcshower LArMCShower_t;
  typedef ::larlite::simch    LArSimCh_t;
  typedef ::larlite::mcstep   LArMCStep_t;
}

//
// Utility functions (geometry, lar properties, etc.)
//
#include "LArUtil/Geometry.h"
namespace supera {
  
  /// Channel number to wire ID
  ::larlite::geo::WireID ChannelToWireID(unsigned int ch);
  
  /// DriftVelocity in cm/us
  double DriftVelocity();
  
  /// Number of time ticks
  unsigned int NumberTimeSamples();
  
  /// Number of channels
  unsigned int Nchannels();
  
  /// Number of planes
  unsigned int Nplanes();
  
  /// Number of wires
  unsigned int Nwires(unsigned int plane);
  
  /// Nearest wire
  unsigned int NearestWire(const TVector3& xyz, unsigned int plane);
  
  /// G4 time to TPC tick (plane 0)
  int TPCG4Time2Tick(double ns);

  /// Tick offset for drift electrons across planes
  double PlaneTickOffset(size_t plane0, size_t plane1);
  
  /// TDC to TPC tick
  double TPCTDC2Tick(double tdc);

  /// Truth position to shifted
  void ApplySCE(double x, double y, double z);
  /// Truth position to shifted
  void ApplySCE(double* xyz);
  /// Truth position to shifted
  void ApplySCE(TVector3& xyz);
  
}

#endif
