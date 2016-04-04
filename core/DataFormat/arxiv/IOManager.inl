#ifndef __LARCV_IOMANAGER_IMP__
#define __LARCV_IOMANAGER_IMP__

#include "IOManager.h"
#include "Base/larbys.h"
#include "ProductMap.h"
#include <algorithm>
namespace larcv {

  template <class T>
  size_t IOManager::register_producer<T>(const std::string& name)
  {
    LARCV_DEBUG() << "start" << std::endl;

    auto const type = ProductType<T>();

    auto in_iter = _key_m.find(name);
    std::string tree_name = ProductName(type) + "_" + name + "_tree";
    std::string tree_desc = name + " tree";
    std::string br_name = ProductName(type) + "_" + name + "_branch";

    LARCV_INFO() << "Requested to register a producer: " << name << " (TTree " << tree_name << ")" << std::endl;
    
    if(in_iter != _key_m.end()) {
      LARCV_INFO() << "... already registered. Returning a registered key " << (*in_iter).second << std::endl;
      return (*in_iter).second;
    }

    _product_ptr_v[_product_ctr] = (EventBase*)(new T);
    _product_type_v[_product_ctr] = type;
    
    const size_t id = _product_ctr-1;
    _key_m.insert(std::make_pair(name,id));

    _product_ctr+=1;

    LARCV_INFO() << "It is a new producer registration (key=" << id << ")" << std::endl;
    
    if(_io_mode != kWRITE) {
      LARCV_INFO() << "kREAD/kBOTH mode: creating an input TChain" << std::endl;
      LARCV_DEBUG() << "Branch name: " << br_name << " data pointer: " << _product_ptr_v[id] << std::endl;
      auto in_tree_ptr = new TChain(tree_name.c_str(),tree_desc.c_str());
      in_tree_ptr->SetBranchAddress(br_name.c_str(), &((T*)(_product_ptr_v.back())));
      _in_tree_v[id] = in_tree_ptr;
      _in_tree_index_v.push_back(kINVALID_SIZE);
    }	
    
    if(_io_mode != kREAD) {
      LARCV_INFO() << "kWRITE/kBOTH mode: creating an output TTree" << std::endl;
      LARCV_DEBUG() << "Branch name: " << br_name << " data pointer: " << _product_ptr_v[id] << std::endl;
      _out_file->cd();
      _out_tree_v[id] = new TTree(tree_name.c_str(),tree_desc.c_str());
      auto out_br_ptr = _out_tree_v[id]->Branch(br_name.c_str(), &((T*)(_product_v[id])));
      LARCV_DEBUG() << "Created TTree @ " << _out_tree_v[id] << " ... TBranch @ " << out_br_ptr << std::endl;

    }

    return id;
  }

  template <class T>
  bool IOManager<T>::read_entry(const size_t index)
  {
    LARCV_DEBUG() << "start" << std::endl;
    if(_io_mode == kWRITE) {
      LARCV_WARNING() << "Nothing to read in kWRITE mode..." << std::endl;
      return false;
    }
    if(!_prepared) {
      LARCV_CRITICAL() << "Cannot be called before initialize()!" << std::endl;
      throw larbys();
    }
    if(index >= _tree_entries) {
      LARCV_ERROR() << "Input only has " << _tree_entries << " entries!" << std::endl;
      return false;
    }
    _tree_index = index;
    LARCV_DEBUG() << "Current tree index: " << _tree_index << std::endl;
    return true;
  }

  template <class T>
  bool IOManager<T>::save_entry()
  {
    LARCV_DEBUG() << "start" << std::endl;

    if(!_prepared) {
      LARCV_CRITICAL() << "Cannot be called before initialize()!" << std::endl;
      throw larbys();
    }

    if(_io_mode == kREAD) {
      LARCV_ERROR() << "Cannot save in READ mode..." << std::endl;
      return false;
    }

    LARCV_INFO() << "Saving new entry " << std::endl;

    for(auto& t : _out_tree_v) {
      if(!t) break;
      LARCV_DEBUG() << "Saving " << t->GetName() << " entry " << t->GetEntries() << std::endl;
      t->Fill();
    }

    _tree_entries += 1;
    if(_io_mode == kWRITE) _tree_index += 1;

    return true;
  }

  template <class T>
  size_t IOManager<T>::producer_id(const std::string& producer) const
  {
    LARCV_DEBUG() << "start" << std::endl;
    auto iter = _key_m.find(producer);
    if(iter == _key_m.end()) {
      return kINVALID_SIZE;      
    }
    return (*iter).second;
  }

  template <class T>
  std::vector<T>& IOManager<T>::get_data(const std::string& producer)
  {
    LARCV_DEBUG() << "start" << std::endl;
    if(producer.empty()) {
      LARCV_CRITICAL() << "Empty producer name (invalid)" << std::endl;
      throw larbys();
    }

    auto id = producer_id(producer);

    if(id == kINVALID_SIZE) {
      if(_io_mode == kREAD) {
	LARCV_ERROR() << "Invalid producer requested:" << producer << std::endl;
	throw larbys();
      }
      id = register_producer(producer);
      for(size_t i=0; i<_tree_entries; ++i) _out_tree_v[id]->Fill();
      LARCV_NORMAL() << "Created TTree " << _out_tree_v[id]->GetName() << " (id=" << id <<") w/ " << _tree_entries << " entries..." << std::endl;
    }
    return get_data(id);
  }

  template <class T>
  std::vector<T>& IOManager<T>::get_data(const size_t id)
  {
    LARCV_DEBUG() << "start" << std::endl;
    if(id >= _product_ctr) {
      LARCV_ERROR() << "Invalid producer ID requested:" << id << std::endl;
      throw larbys();
    }

    if(_io_mode != kWRITE && _tree_index && _tree_index != kINVALID_SIZE &&
       _in_tree_index_v[id] != _tree_index) {
      _in_tree_v[id]->GetEntry(_tree_index);
      _in_tree_index_v[id] =_tree_index;
    }
    return *(_product_v[id]);
  }

  template <class T>
  void IOManager<T>::finalize()
  {
    LARCV_DEBUG() << "start" << std::endl;

    if(_io_mode != kWRITE) {
      LARCV_INFO() << "Deleting input TChains" << std::endl;
      for(auto& t : _in_tree_v) {if(!t) break; delete t;};
    }

    if(_io_mode != kREAD) {
      _out_file->cd();
      for(auto& t : _out_tree_v) {
	if(!t) break;
	LARCV_INFO() << "Writing " << t->GetName() << " with " << t->GetEntries() << " entries" << std::endl;
	t->Write(); 
      }
      LARCV_INFO() << "Closing output file" << std::endl;
      _out_file->Close();
      _out_file = nullptr;
    }

    LARCV_INFO() << "Deleting data pointers" << std::endl;
    for(auto& p : _product_v) { delete p; }

    reset();
  }

  template <class T>
  void IOManager<T>::reset()
  {
    LARCV_DEBUG() << "start" << std::endl;
    _in_tree_v.clear();
    _in_tree_v.resize(1000,nullptr);
    _in_tree_index_v.clear();
    _out_tree_v.clear();
    _out_tree_v.resize(1000,nullptr);
    _product_v.clear();
    _product_v.resize(1000,nullptr);
    _product_ctr = 0;
    _tree_index = 0;
    _tree_entries = 0;
    _prepared = false;
    _out_file_name = "";
    _in_file_v.clear();
    _in_dir_v.clear();
    _key_m.clear();
  }

}
#endif
