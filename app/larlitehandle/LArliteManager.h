#ifndef __LARLITE_MANAGER_H__
#define __LARLITE_MANAGER_H__

/** 
 * \class LArliteManager
 * \brief wrapper for larlite::storage_manager with methods to help sync with an IOManager
 *
 * cycles events, looking for (run,subrun,event) match in each entry. 
 * Past entries are logged, to allow for mapped-lookup.
 *
 */

#include <map>
#include <array>

// larlite
#include "DataFormat/storage_manager.h"

// larcv
#include "Base/larcv_base.h"
#include "DataFormat/IOManager.h"

namespace larcv {

  class LArliteManager : public larlite::storage_manager, larcv::larcv_base {

  public:
    
    LArliteManager( larlite::storage_manager::IOMode_t mode=kUNDEFINED, std::string name="larlite_manager" );
    virtual ~LArliteManager() {};
    
    bool syncEntry( const larcv::IOManager& iolarcv, bool force_reload=false );
    void set_verbosity(::larcv::msg::Level_t level) { ((larcv::larcv_base*)this)->set_verbosity(level); };
    
  protected:

    typedef std::array<int,3> rse_t;
    std::map< rse_t, size_t > m_entry_map;

    rse_t  m_current_rse;
    size_t m_current_entry;
    rse_t  m_last_rse;
    size_t m_last_entry;

  };

}


#endif
