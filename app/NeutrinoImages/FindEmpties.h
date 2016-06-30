/**
 * \file FindEmpties.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class FindEmpties
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __FINDEMPTIES_H__
#define __FINDEMPTIES_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class FindEmpties ... these comments are used to generate
     doxygen documentation!
  */
  class FindEmpties : public ProcessBase {

  public:
    
    /// Default constructor
    FindEmpties(const std::string name="FindEmpties");
    
    /// Default destructor
    ~FindEmpties(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    void reset();

    std::string _image_name;       ///< Image2D producer name
    TTree* _image_tree;                  ///< TTree for analysis later
    unsigned short _image_index;   ///< Image index value
    float  _max_pixel;             ///< Analysis variable: Max pixel value
    int    _pixel_count;           ///< Analysis variable: Pixel count above a certain threshold
    float  _pixel_count_threshold; ///< Analysis variable: Threshold to increment a pixel count

    std::vector<float> _dist_v ; 
    std::vector<float> _pix_intens_v;
    float _max_dist ;

    TTree* _pixel_tree;
    float _pixel_intens;
    int _plane ; ///<

    int _event ;

  };

  /**
     \class larcv::FindEmptiesFactory
     \brief A concrete factory class for larcv::FindEmpties
  */
  class FindEmptiesProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    FindEmptiesProcessFactory() { ProcessFactory::get().add_factory("FindEmpties",this); }
    /// dtor
    ~FindEmptiesProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new FindEmpties(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

