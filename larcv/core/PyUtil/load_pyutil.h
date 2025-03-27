/**
 * \file load_pyutil.h
 *
 * \ingroup PyUtil
 * 
 * \brief Class def header for a class load_pyutil
 *
 * This is a dummy class we can load in order to get ROOT to start
 * loading our libraries when in python environment.
 * 
 * @author kazuhiro
 */

/** \addtogroup PyUtil

    @{*/
#ifndef LOAD_PYUTIL_H
#define LOAD_PYUTIL_H

#include <iostream>

namespace larcv {
  /**
     \class load_pyutil
     User defined class load_pyutil ... these comments are used to generate
     doxygen documentation!
  */
  class load_pyutil{
    
  public:
    
    /// Default constructor
    load_pyutil();

    /// Default destructor
    ~load_pyutil(){}
    
  };
}

#endif
/** @} */ // end of doxygen group 

