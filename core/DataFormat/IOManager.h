/**
 * \file IOManager.h
 *
 * \ingroup LArCV
 * 
 * \brief Class def header for a class IOManager
 *
 * @author drinkingkazu
 */

/** \addtogroup LArCV

    @{*/
#ifndef IOMANAGER_H
#define IOMANAGER_H

#include <iostream>
#include <map>
#include <set>
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include "Base/larcv_base.h"
#include "EventBase.h"
#include "Base/larbys.h"
#include "Base/PSet.h"
#include "ProductMap.h"
namespace larcv {
  /**
     \class IOManager
     User defined class IOManager ... these comments are used to generate
     doxygen documentation!
  */
  class IOManager : public larcv::larcv_base {
    
  public:

    enum IOMode_t { kREAD, kWRITE, kBOTH };

    /// Default constructor
    IOManager(IOMode_t mode=kREAD, std::string name="IOManager");

    /// Configuration PSet construction
    IOManager(const PSet& cfg);
    
    /// Default destructor
    ~IOManager(){}

    IOMode_t io_mode() const { return _io_mode;}
    void reset();
    void add_in_file(const std::string filename, const std::string dirname="");
    void clear_in_file();
    void set_out_file(const std::string name);
    ProducerID_t producer_id(const ProductType_t type, const std::string& producer) const;
    ProductType_t product_type(const size_t id) const;
    void configure(const PSet& cfg);
    bool initialize();
    bool read_entry(const size_t index);
    bool save_entry();
    void finalize();
    void clear_entry();
    void set_id(const size_t run, const size_t subrun, const size_t event);

    size_t get_n_entries() const
    { return (_in_tree_entries ? _in_tree_entries : _out_tree_entries); }
    
    EventBase* get_data(const ProductType_t type, const std::string& producer);
    EventBase* get_data(const ProducerID_t id);

    //
    // Some template class getter for auto-cast
    //
    template <class T> T& get_data(const std::string& producer)
    { return *((T*)(get_data(ProductType<T>(),producer))); }

    template <class T> T& get_data(const ProducerID_t id)
    {
      auto const type = product_type(id);
      if(ProductType<T>() != type) {
	LARCV_CRITICAL() << "Unmatched type (in memory type = " << ProductName(type)
			 << " while specialization type = " << ProductName(ProductType<T>())
			 << std::endl;
	throw larbys();
      }
      return *((T*)(get_data(id)));
    }

    const EventBase& event_id() const { return ( _set_event_id.valid() ? _set_event_id : _event_id ); }

    const std::vector<std::string> producer_list(const ProductType_t type) const
    {
      std::vector<std::string> res;
      for(auto const& key_value : _key_list[type]) res.push_back(key_value.first);
      return res;
    }
    
  private:
    void   set_id();
    void   prepare_input();
    size_t register_producer(const ProductType_t type, const std::string& name);

    IOMode_t    _io_mode;
    bool        _prepared;
    TFile*      _out_file;
    size_t      _in_tree_index;
    size_t      _out_tree_index;
    size_t      _in_tree_entries;
    size_t      _out_tree_entries;
    EventBase   _event_id;
    EventBase   _set_event_id;
    std::string _out_file_name;
    std::vector<std::string>     _in_file_v;
    std::vector<std::string>     _in_dir_v;
    std::vector<std::map<std::string,larcv::ProducerID_t> > _key_list;
    std::vector<TTree*>          _out_tree_v;
    std::vector<TChain*>         _in_tree_v;
    std::vector<size_t>          _in_tree_index_v;
    size_t _product_ctr;
    std::vector<larcv::EventBase*>      _product_ptr_v;
    std::vector<larcv::ProductType_t>   _product_type_v;
    std::vector<std::string> _store_only_name;
    std::vector<larcv::ProductType_t> _store_only_type;
    std::vector<bool> _store_only_bool;
  };

}

#endif

