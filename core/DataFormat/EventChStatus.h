/**
 * \file EventChStatus.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventChStatus
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTCHSTATUS_H
#define EVENTCHSTATUS_H

#include <iostream>
#include "EventBase.h"
#include "ChStatus.h"
#include "DataProductFactory.h"
namespace larcv {
  
  /**
     \class EventChStatus
     User defined class EventChStatus ... these comments are used to generate
     doxygen documentation!
  */
  class EventChStatus : public EventBase {
    
  public:
    
    /// Default constructor
    EventChStatus(){}
    
    /// Default destructor
    virtual ~EventChStatus(){}

    void clear();

    const std::map<larcv::PlaneID_t,larcv::ChStatus>& ChStatusMap() const { return _status_m; }

    const ChStatus& at(PlaneID_t id) const;

    void Insert(PlaneID_t id, const ChStatus& status_v);
    
#ifndef __CINT__
    void Emplace(PlaneID_t id, ChStatus&& status_v);
#endif
    
  private:

    std::map<larcv::PlaneID_t,larcv::ChStatus> _status_m;

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

