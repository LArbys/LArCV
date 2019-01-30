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

namespace larcv {
  /**
     \class IOManager
     User defined class IOManager ... these comments are used to generate
     doxygen documentation!
  */
  template <class T>
  class IOManager : public larcv::larcv_base {
    
  public:

    enum IOMode_t { kREAD, kWRITE, kBOTH };

    /// Default constructor
    IOManager(IOMode_t mode=kREAD) 
      : _io_mode       ( mode          )
      , _prepared      ( false         )
      , _out_file      ( nullptr       )
      , _tree_index    ( 0             )
      , _tree_entries  ( 0             )
      , _out_file_name ( ""            )
      , _in_file_v     ()
      , _in_dir_v      ()
      , _key_m         ()
      , _out_tree_v    ()
      , _in_tree_v     ()
      , _in_tree_index_v ()
      , _product_v     ()
      , _product_ctr   (0)
    { reset(); }
    
    /// Default destructor
    ~IOManager(){}

    void reset();
    void add_in_file(const std::string filename, const std::string dirname="")
    { _in_file_v.push_back(filename); _in_dir_v.push_back(dirname); }
    void set_out_file(const std::string name)
    { _out_file_name = name; }
    size_t producer_id(const std::string& producer) const;
    bool initialize();
    bool read_entry(const size_t index);
    bool save_entry();
    std::vector<T>& get_data(const std::string& producer);
    std::vector<T>& get_data(const size_t id);
    void finalize();

  private:
    void   prepare_input();
    size_t register_producer(const std::string& name);

    IOMode_t    _io_mode;
    bool        _prepared;
    TFile*      _out_file;
    size_t      _tree_index;
    size_t      _tree_entries;
    std::string _out_file_name;
    std::vector<std::string>     _in_file_v;
    std::vector<std::string>     _in_dir_v;
    std::map<std::string,size_t> _key_m;
    std::vector<TTree*>          _out_tree_v;
    std::vector<TChain*>         _in_tree_v;
    std::vector<size_t>          _in_tree_index_v;
    std::vector<std::vector<T>*>              _product_v;
    size_t _product_ctr;
  };

}

#include "IOManager.inl"

#endif

