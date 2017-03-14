/**
 * \file ImageFromPixel2D.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ImageFromPixel2D
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __IMAGEFROMPIXEL2D_H__
#define __IMAGEFROMPIXEL2D_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class ImageFromPixel2D ... these comments are used to generate
     doxygen documentation!
  */
  class ImageFromPixel2D : public ProcessBase {

  public:
    
    /// Default constructor
    ImageFromPixel2D(const std::string name="ImageFromPixel2D");
    
    /// Default destructor
    ~ImageFromPixel2D(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    std::string _pixel2d_producer;
    std::string _ref_image_producer;
    std::string _out_image_producer;

  };

  /**
     \class larcv::ImageFromPixel2DFactory
     \brief A concrete factory class for larcv::ImageFromPixel2D
  */
  class ImageFromPixel2DProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    ImageFromPixel2DProcessFactory() { ProcessFactory::get().add_factory("ImageFromPixel2D",this); }
    /// dtor
    ~ImageFromPixel2DProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new ImageFromPixel2D(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

