/**
 * \file EventPGraph.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventPGraph
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTPGRAPH_H
#define EVENTPGRAPH_H

#include <iostream>
#include "EventBase.h"
#include "PGraph.h"
#include "DataProductFactory.h"
namespace larcv {
  /**
    \class EventPGraph
    Event-wise class to store a collection of larcv::PGraph
  */
  class EventPGraph : public EventBase {
    
  public:
    
    /// Default constructor
    EventPGraph(){}
    
    /// Default destructor
    ~EventPGraph(){}

    /// larcv::PGraph array clearer
    void clear();

    /// larcv::PGraph array const reference getter
    const std::vector<larcv::PGraph>& PGraphArray() const { return _int_v; }

    /// larcv::PGraph array index accessor
    const PGraph& at(size_t id) const;

    /// larcv::PGraph inserter
    void Append(const PGraph& img);

    /// larcv::PGraph array to replace what is stored
    void Set(const std::vector<larcv::PGraph>& int_v);
#ifndef __CINT__
    /// Emplacer for larcv::PGraph
    void Emplace(PGraph&& img);
    /// Emplacer for larcv::PGraph array
    void Emplace(std::vector<larcv::PGraph>&& int_v);
#endif

  private:

    std::vector<larcv::PGraph> _int_v;

  };

  /**
     \class larcv::EventPGraph
     \brief A concrete factory class for larcv::EventPGraph
  */
  class EventPGraphFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventPGraphFactory() { DataProductFactory::get().add_factory(kProductPGraph,this); }
    /// dtor
    ~EventPGraphFactory() {}
    /// create method
    EventBase* create() { return new EventPGraph; }
  };

  
}
#endif
/** @} */ // end of doxygen group 

