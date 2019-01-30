#ifndef __FMWKINTERFACE_CXX__
#define __FMWKINTERFACE_CXX__

#include "FMWKInterface.h"
#include "LArUtil/InvalidWireError.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/TimeService.h"

namespace larcv {
  namespace supera {

    ::larlite::geo::WireID ChannelToWireID(unsigned int ch)
    { return ::larutil::Geometry::GetME()->ChannelToWireID(ch); }

    double DriftVelocity()
    { return ::larutil::LArProperties::GetME()->DriftVelocity(); }

    unsigned int NumberTimeSamples()
    {
      // Till resolved, do not use this function
      throw ::larcv::larbys("NumberTimeSamples function not available!");
      //return ::larutil::DetectorProperties::GetME()->NumberTimeSamples();
      //return 9600;
    }

    unsigned int Nchannels()
    { return ::larutil::Geometry::GetME()->Nchannels(); }

    unsigned int Nplanes()
    { return ::larutil::Geometry::GetME()->Nplanes(); }

    unsigned int Nwires(unsigned int plane)
    { return ::larutil::Geometry::GetME()->Nwires(plane); }

    unsigned int NearestWire(const TVector3& xyz, unsigned int plane)
    {
      /*
      unsigned int res;
      try{
	res = ::larutil::Geometry::GetME()->NearestWire(xyz,plane);
      }catch( ::larutil::InvalidWireError& err){
	res = err.better_wire_number;
      }
      */
      double min_wire=0;
      double max_wire=Nwires(plane)-1;
      double wire = ::larutil::Geometry::GetME()->WireCoordinate(xyz,plane) + 0.5;
      if(wire<min_wire) wire = min_wire;
      if(wire>max_wire) wire = max_wire;

      return (unsigned int)wire;
    }

    int TPCG4Time2Tick(double ns)
    { return ::larutil::TimeService::GetME()->TPCG4Time2Tick(ns); }

    double TPCTDC2Tick(double tdc)
    { return ::larutil::TimeService::GetME()->TPCTDC2Tick(tdc); }      
    
  }
}

#endif
