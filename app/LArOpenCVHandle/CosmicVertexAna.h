/**
 * \file CosmicVertexAna.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class CosmicVertexAna
 *
 * @author jarrett
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __COSMICVERTEXANA_H__
#define __COSMICVERTEXANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImageMaker.h"
#include <TTree.h>
#include <iostream>
namespace larcv {

  /**
     \class ProcessBase
     User defined class CosmicVertexAna ... these comments are used to generate
     doxygen documentation!
  */
  class CosmicVertexAna : public ProcessBase {

  public:
    
    /// Default constructor
    CosmicVertexAna(const std::string name="CosmicVertexAna");
    
    /// Default destructor
    ~CosmicVertexAna(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:

    LArbysImageMaker _LArbysImageMaker;
    
    TTree* _tree;

    int _run;
    int _subrun;
    int _event;
    int _entry;
    int _roid;
    int _vtxid;

    std::string _img2d_prod;
    std::string _pgraph_prod;
    std::string _pcluster_ctor_prod;
    std::string _pcluster_img_prod;
    std::string _reco_roi_prod;
    std::string _thrumu_img_prod;
    std::string _stopmu_img_prod;

    std::vector<int> _num_allpix_v;
    std::vector<int> _num_nottag_v;
    std::vector<int> _num_thrutag_v;
    std::vector<int> _num_stoptag_v;
    std::vector<int> _pts_in_raw_clus_v;
    std::vector<int> _pts_stopmu_ovrlap_v;
    std::vector<int> _pts_thrumu_ovrlap_v;
    std::vector<float> _endCos_v;
    
    ProductType_t _tags_datatype;
    
  };

  /**
     \class larcv::CosmicVertexAnaFactory
     \brief A concrete factory class for larcv::CosmicVertexAna
  */
  class CosmicVertexAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    CosmicVertexAnaProcessFactory() { ProcessFactory::get().add_factory("CosmicVertexAna",this); }
    /// dtor
    ~CosmicVertexAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new CosmicVertexAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

