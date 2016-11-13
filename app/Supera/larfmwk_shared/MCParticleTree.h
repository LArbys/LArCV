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
#include "MCPNode.h"

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

      void RegisterSecondary(const std::vector<U>&, const int time_offset);

      void RegisterSecondary(const std::vector<U>&, const std::vector<W>&, const int time_offset);

      void RegisterSecondary(const std::vector<V>&, const int time_offset);

      void RegisterSecondary(const std::vector<V>&, const std::vector<W>&, const int time_offset);

      void DefinePrimary(const larcv::Vertex& vtx, const larcv::ROI& interaction);

      void DefinePrimaries(const std::vector<U>&, const int time_offset);

      void DefinePrimaries(const std::vector<V>&, const int time_offset);

      void RegisterSecondary(const larcv::Vertex& vtx, const larcv::ROI& secondary);

      void UpdatePrimaryROI();

      std::vector<larcv::supera::InteractionROI_t> GetPrimaryROI() const;
      std::vector<larcv::supera::InteractionROI_t> GetNeutrinoROI() const;

      void clear()
      { 
	_roi_m.clear(); 
	for ( std::vector<MCPNode*>::iterator it=nodelist.begin(); it!=nodelist.end(); it++ ) {
	  delete (*it);
	  (*it) = NULL;
	}
	nodelist.clear();
	idx_primaries.clear();
	primaryids.clear();
	//idx_secondary2primary.clear();
      };

    private:

      Cropper<U,V,W> _cropper;
      std::map<larcv::Vertex,larcv::supera::InteractionROI_t> _roi_m;

      // the particle tree
      std::vector< MCPNode* > nodelist;
      std::vector< int > idx_primaries;
      //std::map< int, int > idx_secondary2primary;
      std::map<larcv::Vertex,int> idx_vertexmap;
      std::map< int, MCPNode* > primaryids;

      std::map<int,double> _min_energy_init_pdg;
      std::map<int,double> _min_energy_deposit_pdg;
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

