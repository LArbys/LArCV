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
#include <TTree.h>
#include <TChain.h>
#include <TFile.h>
#include "Base/larcv_base.h"
#include "EventBase.h"
#include "Base/larbys.h"
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
    IOManager(IOMode_t mode=kREAD) 
      : _io_mode         ( mode          )
      , _prepared        ( false         )
      , _out_file        ( nullptr       )
      , _tree_index      ( 0             )
      , _tree_entries    ( 0             )
      , _out_file_name   ( ""            )
      , _in_file_v       ()
      , _in_dir_v        ()
      , _key_list        ( kProductUnknown )
      , _out_tree_v      ()
      , _in_tree_v       ()
      , _in_tree_index_v ()
      , _product_ctr     (0)
      , _product_ptr_v   ()
      , _product_type_v  ()
    { reset(); }
    
    /// Default destructor
    ~IOManager(){}

    void reset();
    void add_in_file(const std::string filename, const std::string dirname="");
    void set_out_file(const std::string name);
    size_t producer_id(const ProductType_t type, const std::string& producer) const;
    ProductType_t product_type(const size_t id) const;
    bool initialize();
    bool read_entry(const size_t index);
    bool save_entry();
    void finalize();
    void clear_entry();
    void set_id(const size_t run, const size_t subrun, const size_t event);

    size_t get_n_entries() const
    { return _tree_entries; }
    
    EventBase* get_data(const ProductType_t type, const std::string& producer);
    EventBase* get_data(const size_t id);

    //
    // Some template class getter for auto-cast
    //
    template <class T> T& get_data(const std::string& producer)
    { return *((T*)(get_data(ProductType<T>(),producer))); }

    template <class T> T& get_data(const size_t id)
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
    
  private:
    void   set_id();
    void   prepare_input();
    size_t register_producer(const ProductType_t type, const std::string& name);

    IOMode_t    _io_mode;
    bool        _prepared;
    TFile*      _out_file;
    size_t      _tree_index;
    size_t      _tree_entries;
    EventBase   _event_id;
    EventBase   _set_event_id;
    std::string _out_file_name;
    std::vector<std::string>     _in_file_v;
    std::vector<std::string>     _in_dir_v;
    std::vector<std::map<std::string,size_t> > _key_list;
    std::vector<TTree*>          _out_tree_v;
    std::vector<TChain*>         _in_tree_v;
    std::vector<size_t>          _in_tree_index_v;
    size_t _product_ctr;
    std::vector<larcv::EventBase*>      _product_ptr_v;
    std::vector<larcv::ProductType_t>   _product_type_v;
  };

}

#endif

