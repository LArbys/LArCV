/**
 * \file EXTStream.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class EXTStream
 *
 * @author drinkingkazu
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __EXTSTREAM_H__
#define __EXTSTREAM_H__

#include "ImageHolder.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class EXTStream ... these comments are used to generate
     doxygen documentation!
  */
  class EXTStream : public ImageHolder {

  public:
    
    /// Default constructor
    EXTStream(const std::string name="EXTStream");
    
    /// Default destructor
    ~EXTStream(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize(TFile* ana_file);

  private:

    std::string _image_producer;

  };

  /**
     \class larcv::EXTStreamFactory
     \brief A concrete factory class for larcv::EXTStream
  */
  class EXTStreamProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    EXTStreamProcessFactory() { ProcessFactory::get().add_factory("EXTStream",this); }
    /// dtor
    ~EXTStreamProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new EXTStream(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

