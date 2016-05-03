/**
 * \file Compressor.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class Compressor
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __COMPRESSOR_H__
#define __COMPRESSOR_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/Image2D.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class Compressor ... these comments are used to generate
     doxygen documentation!
  */
  class Compressor : public ProcessBase {

  public:
    
    /// Default constructor
    Compressor(const std::string name="Compressor");
    
    /// Default destructor
    ~Compressor(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:

    std::vector<std::string> _image_producer_v;
    std::vector<size_t     > _row_compression_v;
    std::vector<size_t     > _col_compression_v;
    std::vector<Image2D::CompressionModes_t> _mode_v;

  };

  /**
     \class larcv::CompressorFactory
     \brief A concrete factory class for larcv::Compressor
  */
  class CompressorProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    CompressorProcessFactory() { ProcessFactory::get().add_factory("Compressor",this); }
    /// dtor
    ~CompressorProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new Compressor(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

