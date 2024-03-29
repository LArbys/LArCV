/**
 * \file load_pyutil.h
 *
 * \ingroup PyUtil
 * 
 * \brief Class def header for a class load_pyutil
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

     Purpose of this class is to just trigger the loading of the library by calling in python
     
     from larcv import larcv
     larcv.load_pyutil
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

