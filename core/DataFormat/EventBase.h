/**
 * \file EventBase.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventBase
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTBASE_H
#define EVENTBASE_H

#include <iostream>
#include "DataFormatTypes.h"

namespace larcv {
  class IOManager;
  class DataProductFactory;
  /**
     \class EventBase
     User defined class EventBase ... these comments are used to generate
     doxygen documentation!
  */
  class EventBase{
    friend class IOManager;
    friend class DataProductFactory;
  public:
    
    /// Default constructor
    EventBase() : _run    (kINVALID_SIZE)
		, _subrun (kINVALID_SIZE)
		, _event  (kINVALID_SIZE)
    {}

    EventBase(const EventBase& rhs) : _producer(rhs._producer)
				    , _run(rhs._run)
				    , _subrun(rhs._subrun)
				    , _event(rhs._event)
    {}
				      
    /// Default destructor
    virtual ~EventBase(){}

    virtual void clear();

    const std::string & producer() const { return _producer; }
    size_t run()    const { return _run;    }
    size_t subrun() const { return _subrun; }
    size_t event()  const { return _event;  }

    bool valid() const
    { return !(_run == kINVALID_SIZE || _subrun == kINVALID_SIZE || _event == kINVALID_SIZE); }

    inline bool operator==(const EventBase& rhs) const
    { return (_run == rhs.run() && _subrun == rhs.subrun() && _event == rhs.event()); }

    inline bool operator!=(const EventBase& rhs) const
    { return !((*this) == rhs); }

    inline bool operator<(const EventBase& rhs) const
    {
      if(_run < rhs.run()) return true;
      if(_run > rhs.run()) return false;
      if(_subrun < rhs.subrun()) return true;
      if(_subrun > rhs.subrun()) return false;
      if(_event < rhs.event()) return true;
      if(_event > rhs.event()) return false;
      return false;
    }

    /// Formatted string key getter (a string key consists of 0-padded run, subrun, and event id)
    std::string event_key() const;
    
  private:
    std::string _producer; ///< Producer name string
    size_t _run;    ///< LArSoft run number
    size_t _subrun; ///< LArSoft sub-run number
    size_t _event;  ///< LArSoft event number
  };
}

#endif
/** @} */ // end of doxygen group 

