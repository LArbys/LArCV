/**
 * \file EventImage2D.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventImage2D
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENTIMAGE2D_H
#define EVENTIMAGE2D_H

#include <iostream>
#include "EventBase.h"
#include "Image2D.h"
#include "DataProductFactory.h"
namespace larcv {
  
  /**
     \class EventImage2D
     User defined class EventImage2D ... these comments are used to generate
     doxygen documentation!
  */
  class EventImage2D : public EventBase {
    
  public:
    
    /// Default constructor
    EventImage2D(){}
    
    /// Default destructor
    virtual ~EventImage2D(){}

    void clear();

    const std::vector<larcv::Image2D>& Image2DArray() const { return _image_v; }

    const Image2D& at(ImageIndex_t id) const;

    void Append(const Image2D& img);
#ifndef __CINT__
    void Emplace(Image2D&& img);
    void Emplace(std::vector<larcv::Image2D>&& image_v);
#endif
    
  private:

    std::vector<larcv::Image2D> _image_v;

  };

  /**
     \class larcv::EventImage2D
     \brief A concrete factory class for larcv::EventImage2D
  */
  class EventImage2DFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventImage2DFactory() { DataProductFactory::get().add_factory(kProductImage2D,this); }
    /// dtor
    ~EventImage2DFactory() {}
    /// create method
    EventBase* create() { return new EventImage2D; }
  };

}

#endif
/** @} */ // end of doxygen group 

