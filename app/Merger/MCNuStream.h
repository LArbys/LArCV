/**
 * \file MCNuStream.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class MCNuStream
 *
 * @author drinkingkazu
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __MCNUSTREAM_H__
#define __MCNUSTREAM_H__

#include "ImageHolder.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class MCNuStream ... these comments are used to generate
     doxygen documentation!
  */
  class MCNuStream : public ImageHolder {

  public:
    
    /// Default constructor
    MCNuStream(const std::string name="MCNuStream");
    
    /// Default destructor
    ~MCNuStream(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize(TFile* ana_file);

  private:

    std::string _tpc_image_producer;
    std::string _pmt_image_producer;
    std::string _roi_producer;
    std::string _segment_producer;

    double _min_energy_deposit;
    double _min_energy_init;
    double _min_width;
    double _min_height;
  };

  /**
     \class larcv::MCNuStreamFactory
     \brief A concrete factory class for larcv::MCNuStream
  */
  class MCNuStreamProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    MCNuStreamProcessFactory() { ProcessFactory::get().add_factory("MCNuStream",this); }
    /// dtor
    ~MCNuStreamProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new MCNuStream(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

