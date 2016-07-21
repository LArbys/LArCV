/**
 * \file RmEmptyEvts.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class RmEmptyEvts
 *
 * @author ah673
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __RMEMPTYEVTS_H__
#define __RMEMPTYEVTS_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "AnalysisAlg/CalorimetryAlg.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class RmEmptyEvts ... these comments are used to generate
     doxygen documentation!
  */
  class RmEmptyEvts : public ProcessBase {

  public:
    
    /// Default constructor
    RmEmptyEvts(const std::string name="RmEmptyEvts");
    
    /// Default destructor
    ~RmEmptyEvts(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    void reset() ;

    std::string _image_name ;       ///< Image2D producer name
    std::string _roi_name ;       ///< ROI producer name

    TTree* _image_tree;                  ///< TTree for analysis later
    unsigned short _image_index;   ///< Image index value
    int _plane ;
    int _event;
    float _vtx_x ;
    float _vtx_y ;
    float _vtx_z ;
    float _dist_to_wall ;
    float _e_dep ;
    float _e_vis ;
    float _pixel_count ;
    float _child_e_ratio ;
    float _nu_e_ratio ;
    float _worst_ratio ;

    std::vector<float> _child_pdg_v ;
    std::vector<float> _child_ratio_v ;

  };

  /**
     \class larcv::RmEmptyEvtsFactory
     \brief A concrete factory class for larcv::RmEmptyEvts
  */
  class RmEmptyEvtsProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    RmEmptyEvtsProcessFactory() { ProcessFactory::get().add_factory("RmEmptyEvts",this); }
    /// dtor
    ~RmEmptyEvtsProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new RmEmptyEvts(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

