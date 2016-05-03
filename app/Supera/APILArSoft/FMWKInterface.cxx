#ifndef __FMWKINTERFACE_CXX__
#define __FMWKINTERFACE_CXX__

#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "FMWKInterface.h"
#include "lardata/DetectorInfoServices/DetectorPropertiesServiceStandard.h"
#include "lardata/DetectorInfoServices/LArPropertiesServiceStandard.h"
#include "lardata/DetectorInfoServices/DetectorClocksServiceStandard.h"

namespace larcv {
  namespace supera {

    ::geo::WireID ChannelToWireID(unsigned int ch)
    { 
      auto const* geom = ::lar::providerFrom<geo::Geometry>();
      return geom->ChannelToWire(ch).front();
    }

    double DriftVelocity()
    { 
      auto const* detp = ::lar::providerFrom<detinfo::DetectorPropertiesService>();
      return detp->DriftVelocity(); 
    }

    unsigned int NumberTimeSamples()
    {
      throw ::larcv::larbys("NumberTimeSamples function not available!");
      auto const* detp = ::lar::providerFrom<detinfo::DetectorPropertiesService>();
      return detp->NumberTimeSamples(); 
    }

    unsigned int Nchannels()
    {
      auto const* geom = ::lar::providerFrom<geo::Geometry>();
      return geom->Nchannels();
    }

    unsigned int Nplanes()
    { 
      auto const* geom = ::lar::providerFrom<geo::Geometry>();
      return geom->Nplanes();
    }

    unsigned int Nwires(unsigned int plane)
    { 
      auto const* geom = ::lar::providerFrom<geo::Geometry>();
      return geom->Nwires(plane); 
    }

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

    int TPCG4Time2Tick(double ns)
    { 
      auto const* ts = ::lar::providerFrom<detinfo::DetectorClocksServiceStandard>();      
      return ts->TPCG4Time2Tick(ns); 
    }
    
    double TPCTDC2Tick(double tdc)
    { 
      auto const* ts = ::lar::providerFrom<detinfo::DetectorClocksServiceStandard>();
      return ts->TPCTDC2Tick(tdc); 
    }
  }
}

#endif
