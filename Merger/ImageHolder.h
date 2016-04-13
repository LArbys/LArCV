/**
 * \file ImageHolder.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ImageHolder
 *
 * @author drinkingkazu
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __IMAGEHOLDER_H__
#define __IMAGEHOLDER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ROI.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class ImageHolder ... these comments are used to generate
     doxygen documentation!
  */
  class ImageHolder : public ProcessBase {

  public:
    
    /// Default constructor
    ImageHolder(const std::string name="ImageHolder");
    
    /// Default destructor
    virtual ~ImageHolder(){}

    void move(std::vector<larcv::Image2D>&);

    void move(std::vector<larcv::ROI>&);

  private:

    std::vector<larcv::Image2D> _image_v;
    std::vector<larcv::ROI>     _roi_v;
    
  };

}

#endif
/** @} */ // end of doxygen group 

