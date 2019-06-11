#include "LArliteManager.h"

namespace larcv {

  /**
   * constructor. wrapper to larlite::storage_manger class
   *
   * @param[in] mode Possible modes: { kREAD, kWRITE, kBOTH }
   *
   */
  LArliteManager::LArliteManager( larlite::storage_manager::IOMode_t mode, std::string name )
    : larlite::storage_manager(mode), larcv::larcv_base(name)
  {
    m_entry_map.clear();
    m_current_rse   = {-1,-1,-1};
    m_current_entry =  -1;
    m_last_rse   = {-1,-1,-1};
    m_last_entry =  -1;
  }
  
  bool LArliteManager::syncEntry( const larcv::IOManager& iolarcv, bool force_reload ) {
    
    rse_t rse_larcv = { (int)iolarcv.event_id().run(),
                        (int)iolarcv.event_id().subrun(),
                        (int)iolarcv.event_id().event() };

    LARCV_DEBUG() << " sync with larcv "
              << "(" << rse_larcv[0] << "," << rse_larcv[1] << "," << rse_larcv[2] << ")"
              << std::endl;

    // if current_entry<0, we haven't read an entry yet. read first one
    if ( m_current_entry<0 ) {
      LARCV_DEBUG() << " load first entry" << std::endl;
      if ( !go_to(0) )
        return false;
      rse_t first_entry = { (int)run_id(), (int)subrun_id(), (int)event_id() };
      m_entry_map[ first_entry ] = 0;
      if ( m_last_entry<0 ) {
        m_last_entry = m_current_entry;
        m_last_rse   = m_current_rse;
      }
    }

    // check current entry
    m_current_rse   = { (int)run_id(), (int)subrun_id(), (int)event_id() };
    m_current_entry = get_index();
    if (m_current_entry>0 )
      m_current_entry--; // weird larlite index thing I don't understand
    LARCV_DEBUG() << "current index=" << m_current_entry
                  << " (" << m_current_rse[0] << "," << m_current_rse[1] << "," << m_current_rse[2] << ")"
                  << " matches with larcv=" << (m_current_rse==rse_larcv) << std::endl;
    
    if ( m_current_rse==rse_larcv ) {
      // already loaded
      LARCV_DEBUG() << " already same rse" << std::endl;
      return go_to( m_current_entry, force_reload );
    }
    
    bool status = true;

    // debug: dump entry map
    // for ( auto& it : m_entry_map ) {
    //   std::cout << "(" << it.first[0] << "," << it.first[1] << "," << it.first[2] << ") " << it.second << std::endl;
    // }
    
    // look for RSE in past entries
    auto it_rse = m_entry_map.find( rse_larcv );
    if ( it_rse!=m_entry_map.end() ) {
      status = go_to( it_rse->second, force_reload );
      LARCV_DEBUG() <<" found in past rse list: larlite "
                    << "entry=" << it_rse->second
                    << " (" << run_id()  << "," << subrun_id() << "," << event_id() << ")"
                    << std::endl;      
      m_current_rse = rse_larcv;
      m_current_entry  = it_rse->second;
      m_last_rse   = rse_larcv;
      m_last_entry = it_rse->second;
      return status;
    }
    
    // cycle entry until we match new entry
    for ( size_t ientry=m_current_entry; ientry<get_entries(); ientry++ ) {
      status = go_to( ientry, force_reload );
      LARCV_DEBUG() << "loaded larlite entry loop: " << ientry << " index=" << get_index() << std::endl;
      m_current_rse   = { (int)run_id(), (int)subrun_id(), (int)event_id() };
      m_current_entry = ientry;
      m_entry_map[ m_current_rse ] = ientry;
      
      if ( rse_larcv==m_current_rse ) {
        // found it!
        LARCV_DEBUG() << " found larcv rse in entry=" << ientry << std::endl;        
        m_last_rse   = m_current_rse;
        m_last_entry = m_current_entry;
        return status;
      }
    }

    LARCV_DEBUG() << " did not find larcv rse" << std::endl;
    return false;
  }
  
}
