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
    template <class T, class U, class V, class W>
    class MCParticleTree : public larcv::larcv_base {
      
    public:
      
      /// Default constructor
      MCParticleTree() : larcv::larcv_base("MCParticleTree")
		       , _min_energy_init_mcshower(0)
		       , _min_energy_init_mctrack(0)
		       , _min_energy_deposit_mcshower(0)
		       , _min_energy_deposit_mctrack(0)
		       , _min_nplanes(2)
      {}
      
      /// Default destructor
      ~MCParticleTree(){}

      void configure(const Config_t& cfg);

      Cropper<U,V,W>& GetCropper() { return _cropper; }

      void DefinePrimary(const std::vector<T>&);

      void RegisterSecondary(const std::vector<U>&);

      void RegisterSecondary(const std::vector<U>&, const std::vector<W>&);

      void RegisterSecondary(const std::vector<V>&);

      void RegisterSecondary(const std::vector<V>&, const std::vector<W>&);

      void DefinePrimary(const larcv::Vertex& vtx, const larcv::ROI& interaction);

      void RegisterSecondary(const larcv::Vertex& vtx, const larcv::ROI& secondary);

      void UpdatePrimaryROI();

      std::vector<larcv::supera::InteractionROI_t> GetPrimaryROI() const;

      void clear()
      { _roi_m.clear(); }

    private:

      Cropper<U,V,W> _cropper;
      std::map<larcv::Vertex,larcv::supera::InteractionROI_t> _roi_m;

      double _min_energy_init_mcshower;
      double _min_energy_init_mctrack;
      double _min_energy_deposit_mcshower;
      double _min_energy_deposit_mctrack;
      size_t _min_nplanes;
      
    };
  }
}

#include "MCParticleTree.inl"

#endif
/** @} */ // end of doxygen group 

