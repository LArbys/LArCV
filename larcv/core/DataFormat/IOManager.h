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
#include "larcv/core/Base/larcv_base.h"
#include "EventBase.h"
#include "larcv/core/Base/larbys.h"
#include "larcv/core/Base/PSet.h"
#include "ProductMap.h"
namespace larcv {
  /**
    \class IOManager
    LArCV file IO hanlder class: it can read/write LArCV file.
  */
  class IOManager : public larcv::larcv_base {
    
  public:
    /// Three IO modes: read, write, or read-and-write
    enum IOMode_t { kREAD, kWRITE, kBOTH };

    /// What is the tick order on-disk. TickForward, TickBackward. Forward is default. Backward for backwards compatibility.
    enum TickOrder_t { kTickForward, kTickBackward };
    
    /// Default constructor
    IOManager(IOMode_t mode=kREAD, std::string name="IOManager", TickOrder_t tickorder=kTickForward);

    /// Configuration PSet construction so you don't have to call setter functions
    IOManager(const PSet& cfg);
    
    /// Default destructor
    ~IOManager(){}
    /// IO mode accessor
    IOMode_t io_mode() const { return _io_mode;}
    void reset();
    void add_in_file(const std::string filename, const std::string dirname="");
    void clear_in_file();
    void set_out_file(const std::string name);
    ProducerID_t producer_id(const ProductType_t type, const std::string& producer) const;
    ProductType_t product_type(const size_t id) const;
    void configure(const PSet& cfg);
    bool initialize();
    bool read_entry(const size_t index,bool force_reload=false);
    bool save_entry();
    void finalize();
    void clear_entry();
    void set_id(const size_t run, const size_t subrun, const size_t event);
    size_t current_entry() const { return _in_tree_index; }
    
    size_t get_n_entries() const
    { return (_in_tree_entries ? _in_tree_entries : _out_tree_entries); }
    
    EventBase* get_data(const ProductType_t type, const std::string& producer);
    EventBase* get_data(const ProducerID_t id);

    // we provide the option to not automatically clear the write container
    //   after saving an entry. this can help reduce the number of allocs
    //   by allowing us to overwrite values, rather create an entirely new image.
    //   useful if we are creating a large number of images every event
    void donot_clear_product( const ProductType_t type, const std::string& producer );

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

    const EventBase& last_event_id() const { return _last_event_id; }

    const std::vector<std::string> producer_list(const ProductType_t type) const
    {
      std::vector<std::string> res;
      for(auto const& key_value : _key_list[type]) res.push_back(key_value.first);
      return res;
    }

    const std::vector<std::string>& file_list() const
    { return _in_file_v; }

    void specify_data_read( const ProductType_t type, const std::string& name );
    
  private:
    void   set_id();
    void   prepare_input();
    size_t register_producer(const ProductType_t type, const std::string& name);

    IOMode_t    _io_mode;
    TickOrder_t _input_tick_order;
    bool        _prepared;
    TFile*      _out_file;
    size_t      _in_tree_index;
    size_t      _out_tree_index;
    size_t      _in_tree_entries;
    size_t      _out_tree_entries;
    EventBase   _event_id;
    EventBase   _set_event_id;
    EventBase   _last_event_id;
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
    std::vector<std::string> _read_only_name;
    std::vector<larcv::ProductType_t> _read_only_type;
    std::vector<bool> _store_only_bool;
    std::vector<bool> _read_id_bool;
    std::vector<bool> _clear_id_bool;
    //std::map< larcv::ProducerID_t, bool > _image2d_id_wasreversed;
  };

}
#endif

