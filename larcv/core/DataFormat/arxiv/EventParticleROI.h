/**
 * \file EventParticleROI.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventParticleROI
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTPARTICLEROI_H
#define EVENTPARTICLEROI_H

#include <iostream>
#include "EventBase.h"
#include "ParticleROI.h"

namespace larcv {
  /**
     \class EventParticleROI
     User defined class EventParticleROI ... these comments are used to generate
     doxygen documentation!
  */
  class EventParticleROI : public EventBase {
    
  public:
    
    /// Default constructor
    EventParticleROI(){}
    
    /// Default destructor
    ~EventParticleROI(){}

    void clear();

    const std::vector<larcv::ParticleROI>& ParticleROIArray() const { return _part_v; }

    const ParticleROI& at(ParticleROIIndex_t id);

    void Append(const ParticleROI& img);
    void Set(const std::vector<larcv::ParticleROI>& part_v);
#ifndef __CINT__
    void Emplace(ParticleROI&& img);
    void Emplace(std::vector<larcv::ParticleROI>&& part_v);
#endif

  private:

    std::vector<larcv::ParticleROI> _part_v;

  };
}
#endif
/** @} */ // end of doxygen group 

