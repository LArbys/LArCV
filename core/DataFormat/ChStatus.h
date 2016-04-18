/**
 * \file ChStatus.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class ChStatus
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef CHSTATUS_H
#define CHSTATUS_H

#include <iostream>
#include "DataFormatTypes.h"
namespace larcv {
  /**
     \class ChStatus
     User defined class ChStatus ... these comments are used to generate
     doxygen documentation!
  */
  class ChStatus {
    
  public:
    
    /// Default constructor
    ChStatus(){}

#ifndef __CINT__
    ChStatus(PlaneID_t plane, std::vector<short>&&)
      : _status_v(std::move(plane))
      , _plane(plane)
    {}
#endif    
    /// Default destructor
    ~ChStatus(){}

    void  Initialize(size_t nwires, short status = chstatus::kUNKNOWN);

    void  Reset(short status = chstatus::kUNKNOWN);

    void Plane(PlaneID_t p) { _plane = p; }

    void Status(size_t wire, short status);

    PlaneID_t Plane() const { return _plane; }

    short Status(size_t wire) const;

    const std::vector<short>& as_vector() const { return _status_v; }

    std::string dump() const;

  private:
    std::vector<short> _status_v;
    larcv::PlaneID_t _plane;
  };
}

#endif
/** @} */ // end of doxygen group 

