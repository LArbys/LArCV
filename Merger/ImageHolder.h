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
#include "DataFormat/ChStatus.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ROI.h"
#include <map>
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
    void move(std::map<larcv::PlaneID_t,larcv::ChStatus>&);

    const ROI& roi() const { return _roi; }

    void retrieve_id(const EventBase* data)
    { _run=data->run(); _subrun=data->subrun(); _event=data->event(); }
    
    size_t run    () const { return _run;    }
    size_t subrun () const { return _subrun; }
    size_t event  () const { return _event;  }

  protected:

    std::vector<larcv::Image2D>  _image_v;
    std::map<larcv::PlaneID_t,larcv::ChStatus> _chstatus_m;
    ROI _roi;

    size_t _run, _subrun, _event;
  };

}

#endif
/** @} */ // end of doxygen group 

