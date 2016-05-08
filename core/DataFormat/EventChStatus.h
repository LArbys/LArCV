/**
 * \file EventChStatus.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventChStatus
1;2c *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTCHSTATUS_H
#define EVENTCHSTATUS_H

#include <iostream>
#include <map>
#include "EventBase.h"
#include "ChStatus.h"
#include "DataProductFactory.h"
namespace larcv {
  
  /**
    \class EventChStatus
    Event-wise class to store a collection of larcv::ChStatus (one per plane)
  */
  class EventChStatus : public EventBase {
    
  public:
    
    /// Default constructor
    EventChStatus(){}
    
    /// Default destructor
    virtual ~EventChStatus(){}

    /// Clears an array of larcv::ChStatus
    void clear();

    /// Getter for a const reference of core data
    const std::map<larcv::PlaneID_t,larcv::ChStatus>& ChStatusMap() const { return _status_m; }

    /// Get larcv::ChStatus for a specified plane
    const ChStatus& Status(larcv::PlaneID_t id) const;

    /// Insert larcv::ChStatus into a collection
    void Insert(const ChStatus& status);
    
#ifndef __CINT__
    /// Emplace larcv::ChStatus into a collection
    void Emplace(ChStatus&& status);
#endif

    /// Dump contents in string representation
    std::string dump() const;
    
  private:

    std::map< ::larcv::PlaneID_t, ::larcv::ChStatus > _status_m;

  };

  /**
     \class larcv::EventChStatus
     \brief A concrete factory class for larcv::EventChStatus
  */
  class EventChStatusFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventChStatusFactory() { DataProductFactory::get().add_factory(kProductChStatus,this); }
    /// dtor
    ~EventChStatusFactory() {}
    /// create method
    EventBase* create() { return new EventChStatus; }
  };

}

#endif
/** @} */ // end of doxygen group 

