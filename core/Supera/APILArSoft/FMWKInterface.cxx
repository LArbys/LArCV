#ifndef __FMWKINTERFACE_CXX__
#define __FMWKINTERFACE_CXX__

#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "FMWKInterface.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesServiceStandard.h"
#include "lardata/DetectorInfoServices/LArPropertiesServiceStandard.h"
#include "lardata/DetectorInfoServices/DetectorClocksServiceStandard.h"

namespace larcv {
  namespace supera {

    /// Channel number to wire ID
    ::geo::WireID ChannelToWireID(unsigned int ch)
    { 
      auto const* geom = ::lar::providerFrom<geo::Geometry>();
      return geom->ChannelToWire(ch).front();
    }

    /// DriftVelocity in cm/us
    double DriftVelocity()
    { 
      //auto const* larp = ::lar::providerFrom<detinfo::LArPropertiesService>();
      auto const* detp = ::lar::providerFrom<detinfo::DetectorPropertiesService>();
      return detp->DriftVelocity(); 
    }

    /// Number of time ticks
    unsigned int NumberTimeSamples()
    { 
      auto const* detp = ::lar::providerFrom<detinfo::DetectorPropertiesService>();
      return detp->NumberTimeSamples(); 
    }

    /// Number of planes
    unsigned int Nplanes()
    { 
      auto const* geom = ::lar::providerFrom<geo::Geometry>();
      return geom->Nplanes();
    }

    /// Number of wires
    unsigned int Nwires(unsigned int plane)
    { 
      auto const* geom = ::lar::providerFrom<geo::Geometry>();
      return geom->Nwires(plane); 
    }

    /// Nearest wire
    unsigned int NearestWire(const TVector3& xyz, unsigned int plane)
    {
      double min_wire=0;
      double max_wire=Nwires(plane)-1;
      auto const* geom = ::lar::providerFrom<geo::Geometry>();

      double wire = geom->WireCoordinate(xyz[1],xyz[2],plane,0,0) + 0.5;
      if(wire<min_wire) wire = min_wire;
      if(wire>max_wire) wire = max_wire;

      return (unsigned int)wire;
    }

    /// G4 time to TPC tick
    int TPCG4Time2Tick(double ns)
    { 
      auto const* ts = ::lar::providerFrom<detinfo::DetectorClocksServiceStandard>();      
      return ts->TPCG4Time2Tick(ns); 
    }
    
  }
}

#endif
