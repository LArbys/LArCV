#ifndef __FMWKINTERFACE_H__
#define __FMWKINTERFACE_H__

#include "LArUtil/Geometry.h"
#include "FhiclLite/ConfigManager.h"

namespace larcv {
  namespace supera {

    typedef ::fcllite::PSet Config_t;

    /// Channel number to wire ID
    ::larlite::geo::WireID ChannelToWireID(unsigned int ch);

    /// DriftVelocity in cm/us
    double DriftVelocity();

    /// Number of time ticks
    unsigned int NumberTimeSamples();

    /// Number of planes
    unsigned int Nplanes();

    /// Number of wires
    unsigned int Nwires(unsigned int plane);

    /// Nearest wire
    unsigned int NearestWire(const TVector3& xyz, unsigned int plane);

    /// G4 time to TPC tick
    int TPCG4Time2Tick(double ns);
    
  }
}

#endif
