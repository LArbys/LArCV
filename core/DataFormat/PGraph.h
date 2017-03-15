/**
 * \file PGraph.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class PGraph
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef __PGRAPH_H__
#define __PGRAPH_H__

#include <iostream>
#include "DataFormatTypes.h"
#include "ROI.h"
namespace larcv {
  /**
     \class PGraph
     User defined class PGraph ... these comments are used to generate
     doxygen documentation!
  */
  class PGraph {
    
  public:
    
    /// Default constructor
    PGraph(){}

    /// Default destructor
    ~PGraph(){}

    size_t NumParticles() const { return _part_v.size(); }
    
    // Register
    void Add(const larcv::ROI& roi) { _part_v.push_back(roi); }
    void Emplace(larcv::ROI&& roi)  { _part_v.emplace_back(std::move(roi)); }

    const std::vector<larcv::ROI>& Particles() const
    { return _part_v; }

    void Clear();

  private:
    std::vector<larcv::ROI> _part_v;
    std::vector<size_t> _pcluster_idx_v;
  };
}

#endif
/** @} */ // end of doxygen group 

