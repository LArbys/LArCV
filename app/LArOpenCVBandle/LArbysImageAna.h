/**
 * \file LArbysImageAna.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class LArbysImageAna
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __LARBYSIMAGEANA_H__
#define __LARBYSIMAGEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImage.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class LArbysImageAna ... these comments are used to generate
     doxygen documentation!
  */
  class LArbysImageAna : public ProcessBase {

  public:
    
    /// Default constructor
    LArbysImageAna(const std::string name="LArbysImageAna");
    
    /// Default destructor
    ~LArbysImageAna(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);
    
    void finalize();
    
    void SetManager(const::larocv::ImageClusterManager* icm) { _mgr_ptr = icm; }
    
  private:
    
    TTree* _reco_tree;
    
    const ::larocv::ImageClusterManager* _mgr_ptr;
    void Clear();

    /// Unique event keys
    uint _run;
    uint _subrun;
    uint _event;
    
    /// HIP cluster vars
    std::vector<uint> _n_mip_ctors_v;
    std::vector<uint> _n_hip_ctors_v;
    
  };

  /**
     \class larcv::LArbysImageAnaFactory
     \brief A concrete factory class for larcv::LArbysImageAna
  */
  class LArbysImageAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageAnaProcessFactory() { ProcessFactory::get().add_factory("LArbysImageAna",this); }
    /// dtor
    ~LArbysImageAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImageAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

