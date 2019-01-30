/**
 * \file ConfigManager.h
 *
 * \ingroup Base
 * 
 * \brief Class def header for a class ConfigManager
 *
 * @author drinkingkazu
 */

/** \addtogroup Base

    @{*/
#ifndef __LARCVBASE_CONFIGMANAGER_H__
#define __LARCVBASE_CONFIGMANAGER_H__

#include <iostream>
#include "larcv_base.h"
#include "larbys.h"
#include "PSet.h"
#include <set>

namespace larcv {
  /**
     \class ConfigManager
     User defined class ConfigManager ... these comments are used to generate
     doxygen documentation!
  */
  class ConfigManager : public larcv_base {
    
  public:
    
    /// Default constructor
    ConfigManager() : larcv_base("ConfigManager")
    {}
      
    /// Default destructor
    ~ConfigManager(){}
    /// Shared static reference getter
    static const ConfigManager& get() 
    {
      if(!_me) _me = new ConfigManager;
      return *_me;
    }
    /// Adder of configuration from a file
    void AddConfigFile(const std::string cfg_file);
    /// Adder of configuration from parsed string
    void AddConfigString(const std::string cfg_str);
    /// Configuration retrieval method
    const PSet& GetConfig(const std::string cfg);

  private:

    static ConfigManager* _me;
    std::set<std::string> _cfg_files;
    PSet _cfg;
    
  };
}
#endif
/** @} */ // end of doxygen group 

