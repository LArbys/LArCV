/**
 * \file larcv_base.h
 *
 * \ingroup LArCV
 * 
 * \brief Class definition file of larcv_base
 *
 * @author Kazu - Nevis 2015
 */

/** \addtogroup LArCV

    @{*/

#ifndef __LARCV_BASE_H__
#define __LARCV_BASE_H__

#include <vector>
#include "larcv_logger.h"

namespace larcv {
    
  /**
    \class larcv_base
    Framework base class equipped with a logger class
  */
  class larcv_base {
    
  public:
    
    /// Default constructor
    larcv_base(const std::string logger_name="larcv_base")
      : _logger(nullptr)
    { _logger = &(::larcv::logger::get(logger_name)); }
    
    /// Default copy constructor
    larcv_base(const larcv_base &original) : _logger(original._logger) {}
    
    /// Default destructor
    virtual ~larcv_base(){};
    
    /// Logger getter
    inline const larcv::logger& logger() const
    { return *_logger; }
    
    /// Verbosity level
    void set_verbosity(::larcv::msg::Level_t level)
    { _logger->set(level); }

    void set_verbosity( const std::string& level )
    {
      if (level=="debug" )
	set_verbosity( ::larcv::msg::kDEBUG );
      else if (level=="info")
	set_verbosity( ::larcv::msg::kINFO );
      else if (level=="normal")
	set_verbosity( ::larcv::msg::kNORMAL );
      else if (level=="warning")
	set_verbosity( ::larcv::msg::kWARNING );
      else if (level=="error")
	set_verbosity( ::larcv::msg::kERROR );
      else if (level=="critical")
	set_verbosity( ::larcv::msg::kCRITICAL );
      else {
	_logger->send( ::larcv::msg::kCRITICAL ) << "Unrecognized verbosity level name: " << level << std::endl;
      }
    };
	  

    /// Name getter, defined in a logger instance attribute
    const std::string& name() const
    { return logger().name(); }
    
  private:
    
    larcv::logger *_logger;   ///< logger
    
  };
}
#endif

/** @} */ // end of doxygen group
