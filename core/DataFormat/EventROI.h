/**
 * \file EventROI.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventROI
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTROI_H
#define EVENTROI_H

#include <iostream>
#include "EventBase.h"
#include "ROI.h"
#include "DataProductFactory.h"
namespace larcv {
  /**
     \class EventROI
     User defined class EventROI ... these comments are used to generate
     doxygen documentation!
  */
  class EventROI : public EventBase {
    
  public:
    
    /// Default constructor
    EventROI(){}
    
    /// Default destructor
    ~EventROI(){}

    void clear();

    const std::vector<larcv::ROI>& ROIArray() const { return _part_v; }

    const ROI& at(ROIIndex_t id) const;

    void Append(const ROI& img);
    void Set(const std::vector<larcv::ROI>& part_v);
#ifndef __CINT__
    void Emplace(ROI&& img);
    void Emplace(std::vector<larcv::ROI>&& part_v);
#endif

  private:

    std::vector<larcv::ROI> _part_v;

  };

  /**
     \class larcv::EventROI
     \brief A concrete factory class for larcv::EventROI
  */
  class EventROIFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventROIFactory() { DataProductFactory::get().add_factory(kProductROI,this); }
    /// dtor
    ~EventROIFactory() {}
    /// create method
    EventBase* create() { return new EventROI; }
  };

  
}
#endif
/** @} */ // end of doxygen group 

