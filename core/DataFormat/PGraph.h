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
    void Add(const larcv::ROI& roi,size_t pcluster_idx);

    void Emplace(larcv::ROI&& roi,size_t pcluster_idx);

    void Set(const std::vector<larcv::ROI>& roi_v, const std::vector<size_t>& pcluster_idx_v);

    void Emplace(std::vector<larcv::ROI>&& roi_v, std::vector<size_t>&& pcluster_idx_v);
    
    void Clear();

    const std::vector<larcv::ROI>& ParticleArray() const { return _part_v; }
    const std::vector<size_t>& ClusterIndexArray() const { return _pcluster_idx_v; }

  private:
    std::vector<larcv::ROI> _part_v;
    std::vector<size_t> _pcluster_idx_v;
  };
}

#endif
/** @} */ // end of doxygen group 

