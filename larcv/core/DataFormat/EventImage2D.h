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
    Event-wise class to store a collection of larcv::Image2D
  */
  class EventImage2D : public EventBase {
    
  public:
    
    /// Default constructor
    EventImage2D(){}
    
    /// Default destructor
    virtual ~EventImage2D(){}

    /// Clears an array of larcv::Image2D
    void clear();

    /// Const reference getter to an array of larcv::Image2D
    const std::vector<larcv::Image2D>& Image2DArray() const { return _image_v; }

    /// Const reference getter to an array of larcv::Image2D
    const std::vector<larcv::Image2D>& as_vector() const { return _image_v; }

    /// Mutable reference getter to an array of larcv::Image2D
    std::vector<larcv::Image2D>& as_mut_vector() { return _image_v; }
    
    /// larcv::Image2D const reference getter for a specified index number
    const Image2D& at(ImageIndex_t id) const;

    /// larcv::Image2D const reference getter for a specified index number
    Image2D& modimgat(ImageIndex_t id);
    
    /// Inserter into larcv::Image2D array
    void Append(const Image2D& img);
#ifndef __CINT__
    /// Emplace into larcv::Image2D array
    void Emplace(Image2D&& img);
    /// Emplace into larcv::Image2D array
    void Emplace(std::vector<larcv::Image2D>&& image_v);
    /// std::move to retrieve content larcv::Image2D array
    void Move(std::vector<larcv::Image2D>& image_v)
    { image_v = std::move(_image_v); }
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

