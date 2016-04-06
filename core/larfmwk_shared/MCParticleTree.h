/**
 * \file MCParticleTree.h
 *
 * \ingroup APILArLite
 * 
 * \brief Class def header for a class MCParticleTree
 *
 * @author kazuhiro
 */

/** \addtogroup APILArLite

    @{*/
#ifndef MCPARTICLETREE_H
#define MCPARTICLETREE_H

#include <iostream>
#include <vector>
#include <map>
#include "DataFormat/ROI.h"
#include "Base/larcv_base.h"
#include "SuperaTypes.h"
#include "Cropper.h"
namespace larcv {
  namespace supera {

    /**
       \class MCParticleTree
       User defined class MCParticleTree ... these comments are used to generate
       doxygen documentation!
    */
    template <class T, class U, class V>
    class MCParticleTree : public larcv::larcv_base {
      
    public:
      
      /// Default constructor
      MCParticleTree() : larcv::larcv_base("MCParticleTree")
      {}
      
      /// Default destructor
      ~MCParticleTree(){}

      void configure(const Config_t& cfg);

      Cropper<U,V>& GetCropper() { return _cropper; }

      void DefinePrimary(const std::vector<T>&);

      void RegisterSecondary(const std::vector<U>&);

      void RegisterSecondary(const std::vector<V>&);

      void DefinePrimary(const larcv::Vertex& vtx, const larcv::ROI& interaction);

      void RegisterSecondary(const larcv::Vertex& vtx, const larcv::ROI& secondary);

      void UpdatePrimaryROI();

      std::vector<larcv::supera::InteractionROI_t> GetPrimaryROI() const;

      void clear()
      { _roi_m.clear(); }

    private:

      Cropper<U,V> _cropper;
      std::map<larcv::Vertex,larcv::supera::InteractionROI_t> _roi_m;

    };
  }
}

#include "MCParticleTree.inl"

#endif
/** @} */ // end of doxygen group 

