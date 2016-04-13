/**
 * \file BNBNuStream.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class BNBNuStream
 *
 * @author drinkingkazu
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __BNBNUSTREAM_H__
#define __BNBNUSTREAM_H__

#include "ImageHolder.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class BNBNuStream ... these comments are used to generate
     doxygen documentation!
  */
  class BNBNuStream : public ImageHolder {

  public:
    
    /// Default constructor
    BNBNuStream(const std::string name="BNBNuStream");
    
    /// Default destructor
    ~BNBNuStream(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize(TFile* ana_file);

  private:

    std::string _image_producer;
    std::string _roi_producer;
    
    double _min_energy_deposit;
    double _min_energy_init;
    double _min_width;
    double _min_height;
  };

  /**
     \class larcv::BNBNuStreamFactory
     \brief A concrete factory class for larcv::BNBNuStream
  */
  class BNBNuStreamProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    BNBNuStreamProcessFactory() { ProcessFactory::get().add_factory("BNBNuStream",this); }
    /// dtor
    ~BNBNuStreamProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new BNBNuStream(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

