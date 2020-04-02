/**
 * \file EventSparseImage.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class EventSparseImage
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef EVENT_SPARSE_IMAGE_H
#define EVENT_SPARSE_IMAGE_H

#include <iostream>
#include "EventBase.h"
#include "SparseImage.h"
#include "DataProductFactory.h"

namespace larcv {
  
  /**
    \class EventSparseImage
    Event-wise class to store a collection of larcv::SparseImage
  */
  class EventSparseImage : public EventBase {
    
  public:
    
    /// Default constructor
    EventSparseImage(){}
    
    /// Default destructor
    virtual ~EventSparseImage(){}

    /// Clears an array of larcv::SparseImage
    void clear();

    /// Const reference getter to an array of larcv::SparseImage
    const std::vector<larcv::SparseImage>& SparseImageArray() const { return _image_v; }

    /// larcv::SparseImage const reference getter for a specified index number
    const SparseImage& at(ImageIndex_t id) const;

    /// Inserter into larcv::SparseImage array
    void Append(const SparseImage& img);
#ifndef __CINT__
    /// Emplace into larcv::SparseImage array
    void Emplace(SparseImage&& img);
    /// Emplace into larcv::SparseImage array
    void Emplace(std::vector<larcv::SparseImage>&& image_v);
    /// std::move to retrieve content larcv::SparseImage array
    void Move(std::vector<larcv::SparseImage>& image_v)
    { image_v = std::move(_image_v); }
#endif
    
  private:

    std::vector<larcv::SparseImage> _image_v;

  };

  /**
     \class larcv::EventSparseImage
     \brief A concrete factory class for larcv::EventSparseImage
  */
  class EventSparseImageFactory : public DataProductFactoryBase {
  public:
    /// ctor
    EventSparseImageFactory() { DataProductFactory::get().add_factory(kProductSparseImage,this); }
    /// dtor
    ~EventSparseImageFactory() {}
    /// create method
    EventBase* create() { return new EventSparseImage; }
  };

}

#endif
/** @} */ // end of doxygen group 

