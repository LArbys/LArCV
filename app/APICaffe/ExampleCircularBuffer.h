/**
 * \file ExampleCircularBuffer.h
 *
 * \ingroup MultiThreadTools
 * 
 * \brief Class def header for a class ExampleCircularBuffer
 *
 * @author kazuhiro
 */

/** \addtogroup MultiThreadTools

    @{*/
#ifndef EXAMPLECIRCULARBUFFER_H
#define EXAMPLECIRCULARBUFFER_H

#include <iostream>
#include "CircularBufferBase.h"

namespace larcv {

  /**
     \class ExampleCircularBuffer
     User defined class ExampleCircularBuffer ... these comments are used to generate
     doxygen documentation!
  */
  class ExampleCircularBuffer : public larcv::CircularBufferBase<std::vector<double> > {
    
  public:
    
    /// Default constructor
    ExampleCircularBuffer() : larcv::CircularBufferBase<std::vector<double> >()
    { _num_elements = 100; }
    
    /// Default destructor
    ~ExampleCircularBuffer(){ }

    size_t _num_elements;

  protected:

    std::shared_ptr<std::vector<double> > construct() const
    {
      return std::shared_ptr<std::vector<double> >(new std::vector<double>(_num_elements,0.));
    }

    void destruct(std::shared_ptr<std::vector<double> > ptr)
    { ptr->clear(); }

  };
}

#endif
/** @} */ // end of doxygen group 

