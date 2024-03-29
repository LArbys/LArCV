#ifndef __LARCV_IOMANAGER_CXX__
#define __LARCV_IOMANAGER_CXX__

#include "IOManager.h"
#include "larcv/core/Base/larbys.h"
#include "ProductMap.h"
#include "DataProductFactory.h"
#include <algorithm>
#include "EventImage2D.h"
#include "EventClusterMask.h"
#include "EventROI.h"
#include "EventPixel2D.h"

namespace larcv {

  IOManager::IOManager(IOMode_t mode, std::string name, TickOrder_t tickorder)
    : larcv_base(name)
    , _io_mode          ( mode          )
    , _input_tick_order ( tickorder     )
    , _prepared         ( false         )
    , _out_file         ( nullptr       )
    , _in_tree_index    ( 0             )
    , _out_tree_index   ( 0             )
    , _in_tree_entries  ( 0             )
    , _out_tree_entries ( 0             )
    , _out_file_name    ( ""            )
    , _in_file_v        ()
    , _in_dir_v         ()
    , _key_list         ( kProductUnknown )
    , _out_tree_v       ()
    , _in_tree_v        ()
    , _in_tree_index_v  ()
    , _product_ctr      (0)
    , _product_ptr_v    ()
    , _product_type_v   ()
    ,_reverse_all_image2d_products(false)
    ,_reverse_all_roi_products(false)
    ,_reverse_all_pixel2d_products(false)
  { reset(); }

  IOManager::IOManager(const PSet& cfg)
    : larcv_base(cfg.get<std::string>("Name"))
    , _io_mode          ( kREAD         )
    , _prepared         ( false         )
    , _out_file         ( nullptr       )
    , _in_tree_index    ( 0             )
    , _out_tree_index   ( 0             )
    , _in_tree_entries  ( 0             )
    , _out_tree_entries ( 0             )
    , _out_file_name    ( ""            )
    , _in_file_v        ()
    , _in_dir_v         ()
    , _key_list         ( kProductUnknown )
    , _out_tree_v       ()
    , _in_tree_v        ()
    , _in_tree_index_v  ()
    , _product_ctr      (0)
    , _product_ptr_v    ()
    , _product_type_v   ()

  {
    reset();
    configure(cfg);
  }

  void IOManager::add_in_file(const std::string filename, const std::string dirname)
  { _in_file_v.push_back(filename); _in_dir_v.push_back(dirname); }

  void IOManager::clear_in_file()
  { _in_file_v.clear(); }

  void IOManager::set_out_file(const std::string name)
  { _out_file_name = name; }

  ProductType_t IOManager::product_type(const size_t id) const
  {
    if( id > _product_type_v.size() ) {
      LARCV_CRITICAL() << "Product ID " << id << " does not exist... " << std::endl;
      throw larbys();
    }
    return _product_type_v[id];
  }

  void IOManager::configure(const PSet& cfg)
  {
    if(_prepared) throw larbys("Cannot call configure() after initialize()!");

    set_verbosity((msg::Level_t)(cfg.get<unsigned short>("Verbosity",logger().level())));
    _io_mode = (IOMode_t)(cfg.get<unsigned short>("IOMode"));
    _out_file_name = cfg.get<std::string>("OutFileName","");

    // Figure out input files
    _in_file_v.clear();
    _in_file_v = cfg.get<std::vector<std::string> >("InputFiles",_in_file_v);
    _in_dir_v.clear();
    _in_dir_v = cfg.get<std::vector<std::string> >("InputDirs",_in_dir_v);
    if(_in_dir_v.empty()) _in_dir_v.resize(_in_file_v.size(),"");
    if(_in_dir_v.size()!=_in_file_v.size()) {
      LARCV_CRITICAL() << "# of input file (" << _in_file_v.size()
		       << ") != # of input dir (" << _in_dir_v.size() << ")!" << std::endl;
      throw larbys();
    }

    _store_only_name = cfg.get<std::vector<std::string> >("StoreOnlyName",_store_only_name);
    std::vector<unsigned short> store_only_type;
    for(auto const& ptype : _store_only_type) store_only_type.push_back((unsigned short)ptype);
    store_only_type = cfg.get<std::vector<unsigned short> >("StoreOnlyType",store_only_type);
    _store_only_type.clear();
    for(auto const& ptype : store_only_type) {
      if(ptype >= (unsigned short)kProductUnknown) {
	LARCV_CRITICAL() << "StoreOnlyType contains invalid type: "
			 << ptype << " (>=" << kProductUnknown << ")" << std::endl;
	throw larbys();
      }
      _store_only_type.push_back((ProductType_t)(ptype));
    }
    if(_store_only_name.size() != _store_only_type.size()) {
      LARCV_CRITICAL() << "StoreOnlyName and StoreOnlyType has different lengths!" << std::endl;
      throw larbys();
    }

    _read_only_name = cfg.get<std::vector<std::string> >("ReadOnlyName",_read_only_name);
    std::vector<unsigned short> read_only_type;
    for(auto const& ptype : _read_only_type) read_only_type.push_back((unsigned short)ptype);
    read_only_type = cfg.get<std::vector<unsigned short> >("ReadOnlyType",read_only_type);
    _read_only_type.clear();
    for(auto const& ptype : read_only_type) {
      if(ptype >= (unsigned short)kProductUnknown) {
	LARCV_CRITICAL() << "ReadOnlyType contains invalid type: "
			 << ptype << " (>=" << kProductUnknown << ")" << std::endl;
	throw larbys();
      }
      _read_only_type.push_back((ProductType_t)(ptype));
    }
    if(_read_only_name.size() != _read_only_type.size()) {
      LARCV_CRITICAL() << "ReadOnlyName and ReadOnlyType has different lengths!" << std::endl;
      throw larbys();
    }

    _input_tick_order = (!cfg.get<bool>("TickBackward",false)) ? kTickForward :  kTickBackward;

    // tick-reversal products
    // we first check if "all" is provided
    std::vector< std::string >* reverse_list_v[3] = { &_reverse_image2d_products,
                                                      &_reverse_roi_products,
                                                      &_reverse_pixel2d_products };
    bool* reverse_all[3] = { &_reverse_all_image2d_products,
                             &_reverse_all_roi_products,
                             &_reverse_all_pixel2d_products };
    std::vector< std::string >  reverse_keynames_v = { "ReverseImage2DList",
                                                       "ReverseROIList",
                                                       "ReversePixel2DList" };
    for (int i=0; i<3; i++ ) {
      try {
        std::string isall = cfg.get<std::string >(reverse_keynames_v[i]);
        if (isall=="ALL" || isall=="all" || isall=="All") {
          (*reverse_all[i]) = true;
        }
        else {
          throw std::runtime_error("not a proper command");
        }
      }
      catch (...) {
        // get the list
        (*reverse_list_v[i]) = cfg.get<std::vector<std::string> >(reverse_keynames_v[i],std::vector<std::string>());
      }
    }
    
  }// end of configure(const larcv::PSet&)

  bool IOManager::initialize()
  {
    LARCV_DEBUG() << "start" << std::endl;

    if(_io_mode != kREAD) {

      if(_out_file_name.empty()) throw larbys("Must set output file name!");
      LARCV_INFO() << "Opening an output file: " << _out_file_name << std::endl;
      _out_file = TFile::Open(_out_file_name.c_str(),"RECREATE");

    }

    if(_io_mode != kWRITE) {
      prepare_input();
      if(!_in_tree_entries) {
	LARCV_ERROR() << "Found 0 entries from input files..." << std::endl;
	return false;
      }
      LARCV_NORMAL() << "Prepared input with " << _in_tree_entries << " entries..." << std::endl;
      _read_id_bool.clear();
      _read_id_bool.resize(_product_ctr,true);
    }


    // Now handle "store-only" configuration
    _store_only_bool.clear();
    if(_io_mode != kREAD && _store_only_type.size()) {
      std::vector<size_t> store_only_id;
      for(size_t i=0; i<_store_only_type.size(); ++i)
	store_only_id.push_back(register_producer(_store_only_type[i],_store_only_name[i]));
      _store_only_bool.resize(_product_ctr,false);
      if ( _product_ctr>_read_id_bool.size() ) // append to read-in counters
	_read_id_bool.resize(_product_ctr,false);
      for(auto const& id : store_only_id) _store_only_bool.at(id) = true;
    }

    _in_tree_index = 0;
    _out_tree_index = 0;
    _prepared = true;

    // clear id: flag to prevent clearing product upon writing to disk
    if ( _product_ctr>=_clear_id_bool.size() )
      _clear_id_bool.resize(_product_ctr+10,true);

    return true;
  }

  size_t IOManager::register_producer(const ProductType_t type, const std::string& name )
  {
    LARCV_DEBUG() << "start" << std::endl;

    if(type == kProductUnknown) {
      LARCV_CRITICAL() << "Cannot register a producer " << name << " for kProductUnknown!" << std::endl;
      throw larbys();
    }

    auto& key_m = _key_list[type];

    auto in_iter = key_m.find(name);
    std::string tree_name = std::string(ProductName(type)) + "_" + name + "_tree";
    std::string tree_desc = name + " tree";
    std::string br_name   = std::string(ProductName(type)) + "_" + name + "_branch";

    LARCV_INFO() << "Requested to register a producer: " << name << " (TTree " << tree_name << ")" << std::endl;

    if(in_iter != key_m.end()) {
      LARCV_INFO() << "... already registered. Returning a registered key " << (*in_iter).second << std::endl;
      return (*in_iter).second;
    }

    _product_ptr_v[_product_ctr]  = (EventBase*)(DataProductFactory::get().create(type,name));
    _product_type_v[_product_ctr] = type;

    if ( _product_ctr>=_clear_id_bool.size() ) {
      LARCV_DEBUG() << "increase clear_id_bool vector (size=" << _clear_id_bool.size() << ") request=" << _product_ctr << std::endl;
      _clear_id_bool.resize(_product_ctr+10,true);
    }
    _clear_id_bool[_product_ctr]  = true;

    const ProducerID_t id = _product_ctr;
    key_m.insert(std::make_pair(name,id));

    _product_ctr+=1;

    LARCV_INFO() << "It is a new producer registration (key=" << id << ")" << std::endl;

    // Set event ID
    if(_event_id.valid()) {
      _product_ptr_v[id]->_run    = _event_id.run();
      _product_ptr_v[id]->_subrun = _event_id.subrun();
      _product_ptr_v[id]->_event  = _event_id.event();
    }else if(_set_event_id.valid()) {
      _product_ptr_v[id]->_run    = _set_event_id.run();
      _product_ptr_v[id]->_subrun = _set_event_id.subrun();
      _product_ptr_v[id]->_event  = _set_event_id.event();
    }

    if(_io_mode != kWRITE) {
      LARCV_INFO() << "kREAD/kBOTH mode: creating an input TChain" << std::endl;
      LARCV_DEBUG() << "Branch name: " << br_name << " data pointer: " << _product_ptr_v[id] << std::endl;
      auto in_tree_ptr = new TChain(tree_name.c_str(),tree_desc.c_str());
      in_tree_ptr->SetBranchAddress(br_name.c_str(), &(_product_ptr_v[id]));
      _in_tree_v[id] = in_tree_ptr;
      _in_tree_index_v.push_back(kINVALID_SIZE);

      // We determine if the product has been marked to have tick-order reversed
      if ( _input_tick_order==kTickBackward ) {
        
 	bool reverseme = false;
 	if ( type==kProductImage2D ) {
 	  if ( _reverse_all_image2d_products ) {
 	    reverseme = true;
 	  }
          else {
 	    for (auto const& reverse_prod : _reverse_image2d_products ) {
 	      if ( reverse_prod==name ) {
 		reverseme = true;
 		LARCV_DEBUG() << " input image2d [" << name << "] is in tick-backward order, will be reveresed" << std::endl;
 		break;
 	      }
 	    }
 	  }
 	}
 	else if ( type==kProductROI ) {
 	  if ( _reverse_all_roi_products ) reverseme = true;
 	  else {
 	    for (auto const& reverse_prod : _reverse_roi_products ) {
 	      if ( reverse_prod==name ) {
 		LARCV_DEBUG() << " input ROI [" << name << "] is in tick-backward order, will be reveresed" << std::endl;	      
 		reverseme = true;
 		break;
 	      }
 	    }
 	  }
 	}
 	else if ( type==kProductPixel2D ) {
 	  if ( _reverse_all_pixel2d_products ) reverseme = true;
 	  else {
 	    for (auto const& reverse_prod : _reverse_pixel2d_products ) {
 	      if ( reverse_prod==name ) {
 		LARCV_DEBUG() << " input Pixel2D [" << name << "] is in tick-backward order, will be reveresed" << std::endl;	      
 		reverseme = true;
 		break;
 	      }
 	    }
 	  }
 	}
         
 	if ( reverseme ) {
 	  // register product ID to be reversed
 	  _reverse_productid.insert( id );
 	}
 	
      }//if tickbackward
      
    }

    if(_io_mode != kREAD) {
      LARCV_INFO() << "kWRITE/kBOTH mode: creating an output TTree" << std::endl;
      LARCV_DEBUG() << "Branch name: " << br_name << " data pointer: " << _product_ptr_v[id] << "(" << id <<"/"<<_product_ptr_v.size()<<")"<<std::endl;
      _out_file->cd();
      _out_tree_v[id] = new TTree(tree_name.c_str(),tree_desc.c_str());
      auto out_br_ptr = _out_tree_v[id]->Branch(br_name.c_str(), &(_product_ptr_v[id]));
      LARCV_DEBUG() << "Created TTree @ " << _out_tree_v[id] << " ... TBranch @ " << out_br_ptr << std::endl;
    }

    return id;
  }

  void IOManager::prepare_input()
  {
    LARCV_DEBUG() << "start" << std::endl;
    if(_product_ctr) {
      LARCV_CRITICAL() << "Cannot call prepare_input before calling reset()!" << std::endl;
      throw larbys();
    }

    LARCV_INFO() << "Start inspecting " << _in_file_v.size() << "files" << std::endl;
    for(size_t i=0; i<_in_file_v.size(); ++i) {

      auto const& fname = _in_file_v[i];
      auto const& dname = _in_dir_v[i];

      TFile *fin = TFile::Open(fname.c_str(),"READ");
      if(!fin) {
	LARCV_CRITICAL() << "Open attempt failed for a file: " << fname << std::endl;
	throw larbys();
      }

      LARCV_NORMAL() << "Opening a file in READ mode: " << fname << std::endl;
      TDirectoryFile* fin_dir = 0;
      if(dname.empty()) fin_dir = fin;
      else {
	TObject* obj = fin->Get(dname.c_str());
	if(!obj || std::string(obj->ClassName())!="TDirectoryFile") {
	  LARCV_CRITICAL() << "Could not locate TDirectoryFile: " << dname << std::endl;
	  throw larbys();
	}
	fin_dir = (TDirectoryFile*)obj;
      }

      TList* key_list = fin_dir->GetListOfKeys();
      TIter key_iter(key_list);
      std::set<std::string> processed_object;
      while(1) {
	TObject* obj = key_iter.Next();
	if(!obj) break;
	if(processed_object.find(obj->GetName()) != processed_object.end()) continue;
	obj = fin_dir->Get(obj->GetName());
	LARCV_DEBUG() << "Found object " << obj->GetName() << " (type=" << obj->ClassName() << ")" << std::endl;
	processed_object.insert(obj->GetName());

	if(std::string(obj->ClassName())!="TTree") {
	  LARCV_DEBUG() << "Skipping " << obj->GetName() << " ... (not TTree)" << std::endl;
	  continue;
	}

	std::string obj_name = obj->GetName();

	char c[2] = "_";
	if(obj_name.find_first_of(c) > obj_name.size() ||
	   obj_name.find_first_of(c) == obj_name.find_last_of(c)) {
	    LARCV_INFO() << "Skipping " << obj->GetName() << " ... (not LArCV TTree)" << std::endl;
	    continue;
	}

	std::string type_name( obj_name.substr(0,obj_name.find_first_of(c)) );
	std::string suffix( obj_name.substr(obj_name.find_last_of(c)+1, obj_name.size()-obj_name.find_last_of(c)) );
	std::string producer_name( obj_name.substr(obj_name.find_first_of(c)+1,obj_name.find_last_of(c)-obj_name.find_first_of(c)-1) );

	if(suffix != "tree") {
	  LARCV_INFO() << "Skipping " << obj->GetName() << " ... (not LArCV TTree)" << std::endl;
	  continue;
	}

	larcv::ProductType_t type = larcv::kProductUnknown;
	for(size_t i=0; i<((size_t)(larcv::kProductUnknown)); ++i) {
	  auto temp_type = (larcv::ProductType_t)i;
	  if(type_name != ProductName(temp_type)) {
	    LARCV_DEBUG() << "TTree " << obj->GetName() << " != type " << ProductName(temp_type) << std::endl;
	    continue;
	  }
	  type = temp_type;
	  break;
	}
	if(type == larcv::kProductUnknown) {
	  LARCV_INFO() << "TTree " << obj->GetName() << " not LArCV data product TTree. Skipping..." << std::endl;
	  continue;
	}

	// If read-only is specified and not in a list, skip
	if(_read_only_name.size()) {
	  bool skip=true;
	  for(auto const& read_type : _read_only_type) {
	    if(read_type != type) continue;
	    for(auto const& read_name : _read_only_name) {
	      if(read_name == producer_name) {
		skip=false;
		break;
	      }
	    }
	  }
	  if(skip) {
	    LARCV_NORMAL() << "Skipping: producer=" << producer_name << " type= " << ProductName(type) << std::endl;
	    continue;
	  }
	  LARCV_INFO() << "Not skipping: producer=" << producer_name << " type= " << ProductName(type) << std::endl;
	}

	auto id = register_producer(type,producer_name);
	LARCV_INFO() << "Registered: producer=" << producer_name << " Key=" << id << std::endl;
	_in_tree_v[id]->AddFile(fname.c_str());
      }
    }

    if(!_in_tree_v.front()) {
      _in_tree_entries = 0;
      return;
    }

    // Get tree entries
    _in_tree_entries = kINVALID_SIZE;
    for(auto const& t : _in_tree_v) {
      if(!t) break;
      size_t tmp_entries = t->GetEntries();
      LARCV_INFO() << "TTree " << t->GetName() << " has " << tmp_entries << " entries" << std::endl;
      if(_in_tree_entries == kINVALID_SIZE) _in_tree_entries = tmp_entries;
      else _in_tree_entries = (_in_tree_entries < tmp_entries ? _in_tree_entries : tmp_entries);
    }

  }

  bool IOManager::read_entry(const size_t index,bool force_reload)
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
    if(index >= _in_tree_entries) {
      LARCV_ERROR() << "Input only has " << _in_tree_entries << " entries!" << std::endl;
      return false;
    }
    if(_in_tree_index != index) {
      _in_tree_index = index;
      _event_id.clear();
      _set_event_id.clear();
    }else if(force_reload){
      _in_tree_index = index;
      for(auto& v : _in_tree_index_v) v = kINVALID_SIZE;
      _event_id.clear();
      _set_event_id.clear();
    }
    LARCV_DEBUG() << "Current input tree index: " << _in_tree_index << std::endl;
    return true;
  }

  bool IOManager::save_entry()
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

    // in kBOTH mode make sure all TTree entries are read-in
    if(_io_mode == kBOTH) {
      for(size_t id=0; id<_in_tree_index_v.size(); ++id) {
	if(_store_only_bool.size() && (id >= _store_only_bool.size() || !_store_only_bool[id])) continue;
	if(_in_tree_index_v[id] == _in_tree_index) continue;
	get_data(id);
      }
    }

    LARCV_INFO() << "Saving new entry " << std::endl;

    set_id();

    if(_store_only_bool.empty()) {

      for(auto& p : _product_ptr_v)  {
	if(!p) break;
	if(!p->valid()) {
	  LARCV_CRITICAL() << "Product by a producer " << p->producer()
			   << " has an invalid event id: ("
			   << p->run() << "," << p->subrun() << "," << p->event() << ")" << std::endl;
	  throw larbys("Must set an event ID to store!");
	}
      }

      for(size_t i=0; i<_out_tree_v.size(); ++i) {
	auto& t = _out_tree_v[i];
	auto& p = _product_ptr_v[i];
	if(!t) break;
	LARCV_DEBUG() << "Saving " << t->GetName() << " entry " << t->GetEntries() << std::endl;
	t->Fill();
	if ( _clear_id_bool[i] )
	  p->clear();
      }

    }else{

      for(size_t i=0; i<_store_only_bool.size(); ++i) {
	auto const& p = _product_ptr_v[i];
	if(!_store_only_bool[i]) {
	  if ( _clear_id_bool[i] )
	    p->clear();
	  continue;
	}
	if(!p->valid()) {
	  LARCV_CRITICAL() << "Invalid event id: (" << p->run() << "," << p->subrun() << "," << p->event() << ")" << std::endl;
	  throw larbys("Must set an event ID to store!");
	}
      }

      for(size_t i=0; i<_store_only_bool.size(); ++i) {
	if(!_store_only_bool[i]) continue;
	auto& t = _out_tree_v[i];
	auto& p = _product_ptr_v[i];
	LARCV_DEBUG() << "Saving " << t->GetName() << " entry " << t->GetEntries() << std::endl;
	t->Fill();
	if ( _clear_id_bool[i] )
	  p->clear();
      }
    }

    clear_entry();

    _out_tree_entries += 1;
    _out_tree_index += 1;

    return true;
  }

  void IOManager::clear_entry()
  {
    int pid = -1;
    for(auto& p : _product_ptr_v) {
      pid++;
      if(!p) break;
      if ( _clear_id_bool[pid] ) {
	LARCV_DEBUG() << "clear product " << p->producer() << "[" << pid << "] of " << _product_ptr_v.size() << std::endl;
	p->clear();
      }
      else {
	LARCV_DEBUG() << "prevented from clearing product " << p->producer() << "[" << pid << "] of " << _product_ptr_v.size() << std::endl;
      }
    }
    if(_set_event_id.valid()) {
      LARCV_DEBUG() << "Set _last_event_id to externally set values:"
		    << " run = " << _set_event_id.run()
		    << " subrun = " << _set_event_id.subrun()
		    << " event = " << _set_event_id.event() << std::endl;
      _last_event_id = _set_event_id;
    }
    else {
      LARCV_DEBUG() << "Set _last_event_id to inherited values:"
		    << " run = " << _event_id.run()
		    << " subrun = " << _event_id.subrun()
		    << " event = " << _event_id.event() << std::endl;
      _last_event_id = _event_id;
    }
    _event_id.clear();
    _set_event_id.clear();
    // reset reverse-tick flags
    //for ( auto& it_reversed : _image2d_id_wasreversed )
    //it_reversed.second = true;
  }

  ProducerID_t IOManager::producer_id(const ProductType_t type, const std::string& producer) const
  {
    LARCV_DEBUG() << "start" << std::endl;
    if(producer.empty()) {
      LARCV_CRITICAL() << "Empty producer name (invalid)" << std::endl;
      throw larbys();
    }
    if(type==kProductUnknown) {
      LARCV_CRITICAL() << "Queried kProductUnknown type!" << std::endl;
      throw larbys();
    }
    // for
    // LARCV_DEBUG() << _key_list  <<  std::endl;

    auto& m = _key_list[type];
    auto iter = m.find(producer);

    if(iter == m.end()) {
      return kINVALID_PRODUCER;
    }
    return (*iter).second;
  }

  EventBase* IOManager::get_data(const ProductType_t type, const std::string& producer)
  {
    LARCV_DEBUG() << "start" << std::endl;
    auto id = producer_id(type,producer);

    if(id == kINVALID_SIZE) {
      if(_io_mode == kREAD) {
	LARCV_ERROR() << "Invalid producer requested: " << producer << " for " << ProductName(type) << std::endl;
	throw larbys();
      }
      id = register_producer(type,producer);
      //for(size_t i=0; i<_in_tree_index; ++i) _out_tree_v[id]->Fill();
      LARCV_NORMAL() << "Created TTree " << _out_tree_v[id]->GetName() << " (id=" << id <<") w/ index " << _in_tree_index << " ..." << std::endl;
    }
    return get_data(id);
  }

  EventBase* IOManager::get_data(const size_t id)
  {
    LARCV_DEBUG() << "start" << std::endl;

    if(id >= _product_ctr) {
      LARCV_ERROR() << "Invalid producer ID requested:" << id << std::endl;
      throw larbys();
    }

    if(_io_mode != kWRITE && _in_tree_index != kINVALID_SIZE &&
       _in_tree_index_v[id] != _in_tree_index && _read_id_bool[id] ) {

      LARCV_DEBUG() << "Reading in TTree " << _in_tree_v[id]->GetName() << " index " << _in_tree_index << std::endl;

      _in_tree_v[id]->GetEntry(_in_tree_index);
      _in_tree_index_v[id] =_in_tree_index; // marks that we've already asked for the entry


      auto& ptr = _product_ptr_v[id];
      // retrieve event_id if not yet done
      if(!_event_id.valid()) {
	LARCV_INFO() << "Setting event id (" << ptr->run() << "," << ptr->subrun() << "," << ptr->event() << ")" << std::endl;
	_event_id = (EventBase)(*ptr);
      }else if(ptr->valid() && _event_id != *ptr) {
	LARCV_CRITICAL() << "Event alignment error (run,subrun,event) detected: "
			 << "Current (" << _event_id.run() << "," << _event_id.subrun() << "," << _event_id.event() << ") vs. "
			 << "Read-in (" << ptr->run() << "," << ptr->subrun() << "," << ptr->event() << ")" << std::endl;
	throw larbys();
      }

      // if image2d and input is expected to be in reverse-tick order AND we haven't flipped it yet,
      // then we flip the row data
      if ( _input_tick_order==IOManager::kTickBackward ) {

	if ( _reverse_productid.find(id)!=_reverse_productid.end() ) {

	  if (  product_type(id)==kProductImage2D ) {

	    LARCV_DEBUG() << "Input image2d [" << _in_tree_v[id]->GetName() << "] expected in tick-backward order." << std::endl;
	    LARCV_DEBUG() << "Reverse to tick-forward. Move origin to be w.r.t bottom left" << std::endl;
	    EventImage2D* evimg = (EventImage2D*)ptr;
	    std::vector<Image2D> img_v;
	    evimg->Move(img_v);
	    for ( auto& img2d : img_v ) {
	      img2d.reverseTimeOrder();
	    }
	    evimg->Emplace(std::move(img_v));
	  }
	  else if (  product_type(id)==kProductROI ) {

	    // make copy
	    EventROI* evroi = (EventROI*)ptr;
	    std::vector<larcv::ROI> roi_v = evroi->ROIArray();
	    for ( auto& roi : roi_v ) {
	      // copy of BB
	      std::vector<larcv::ImageMeta> bb_v = roi.BB();
	      for ( auto& bb : bb_v ) {
		bb.reset_origin( bb.min_x(), bb.min_y() - bb.height() );
	      }
	      roi.SetBB(bb_v); // replace
	    }
	    evroi->Set( roi_v  );
	  }
          else if ( product_type(id)==kProductPixel2D ) {
            ((EventPixel2D*)ptr)->reverseTickOrder();
          }
	    
	}
      }//end if tick-backward flag on
      
    }// READ NEW INPUT ENTRY

    return _product_ptr_v[id];
  }

  void IOManager::set_id(const size_t run, const size_t subrun, const size_t event) {

    if(_io_mode == kREAD) {
      LARCV_CRITICAL() << "Cannot change event id in kREAD mode" << std::endl;
      throw larbys();
    }

    EventBase tmp;
    tmp._run    = run;
    tmp._subrun = subrun;
    tmp._event  = event;

    LARCV_INFO() << "Request to set event id: " << tmp.event_key() << std::endl;

    if(_set_event_id.valid() && _set_event_id != tmp)
      LARCV_INFO() << "Force setting (run,subrun,event) ID as (" << run << "," << subrun << "," << event << ")" << std::endl;

    _set_event_id = tmp;

  }

  void IOManager::set_id() {
    LARCV_DEBUG() << "start" << std::endl;

    if(_io_mode == kREAD) return;

    if(_set_event_id.valid())

      _event_id = _set_event_id;

    LARCV_INFO() << "Setting event id for output trees: " << _event_id.event_key() << std::endl;

    for(size_t i=0; i<_product_ptr_v.size(); ++i) {

      auto& p = _product_ptr_v[i];
      if(!p) break;
      LARCV_DEBUG() << "Updating event id for product " << ProductName(_product_type_v[i]) << " by " << p->producer() << std::endl;
      if( (*p) != _event_id ) {
	if(p->valid()) {
	  LARCV_WARNING() << "Override event id for product " << ProductName(_product_type_v[i])
			  << " by " << p->producer()
			  << " from (" << p->run() << "," << p->subrun() << "," << p->event() << ")"
			  << " to (" << _event_id.run() << "," << _event_id.subrun() << "," << _event_id.event() << ")" << std::endl;
	}
	p->_run = _event_id.run();
	p->_subrun = _event_id.subrun();
	p->_event = _event_id.event();
      }

    }
  }

  void IOManager::finalize()
  {
    LARCV_DEBUG() << "start" << std::endl;

    if(_io_mode != kWRITE) {
      LARCV_INFO() << "Deleting input TChains" << std::endl;
      for(auto& t : _in_tree_v) {if(!t) break; delete t;};
    }

    if(_io_mode != kREAD) {
      _out_file->cd();
      if(_store_only_bool.empty()) {
	for(auto& t : _out_tree_v) {
	  if(!t) break;
	  LARCV_NORMAL() << "Writing " << t->GetName() << " with " << t->GetEntries() << " entries" << std::endl;
	  t->Write();
	}
      }else{
	for(size_t i=0; i<_store_only_bool.size(); ++i) {
	  if(!_store_only_bool[i]) continue;
	  auto& t = _out_tree_v[i];
	  LARCV_NORMAL() << "Writing " << t->GetName() << " with " << t->GetEntries() << " entries" << std::endl;
	  t->Write();
	}
      }
      LARCV_NORMAL() << "Closing output file" << std::endl;
      _out_file->Close();
      _out_file = nullptr;
    }

    LARCV_INFO() << "Deleting data pointers" << std::endl;
    for(auto& p : _product_ptr_v) { delete p; }

    reset();
  }

  void IOManager::reset()
  {
    LARCV_DEBUG() << "start" << std::endl;
    _event_id.clear();
    _set_event_id.clear();
    _in_tree_v.clear();
    _in_tree_v.resize(1000,nullptr);
    _in_tree_index_v.clear();
    _out_tree_v.clear();
    _out_tree_v.resize(1000,nullptr);
    _product_ptr_v.clear();
    _product_ptr_v.resize(1000,nullptr);
    _product_type_v.clear();
    _product_type_v.resize(1000,kProductUnknown);
    _product_ctr = 0;
    _in_tree_index = 0;
    _out_tree_index = 0;
    _in_tree_entries = 0;
    _prepared = false;
    _out_file_name = "";
    _in_file_v.clear();
    _in_dir_v.clear();
    for(auto& m : _key_list) m.clear();
    _clear_id_bool.clear();
    _clear_id_bool.resize(1000,true);
    _reverse_productid.clear();
    _reverse_all_image2d_products = false;
    _reverse_all_roi_products = false;
    _reverse_all_pixel2d_products = false;
    _reverse_image2d_products.clear();
    _reverse_roi_products.clear();
    _reverse_pixel2d_products.clear();
  }

  void IOManager::donot_clear_product( const ProductType_t type, const std::string& producer ) {
    // We do not clear the product upon writing
    // Warning!! user must manager this now!!
    LARCV_DEBUG() << "IOManager no longer responsible for clearing output vectors for producer=" << producer << std::endl;
    ProducerID_t pid = producer_id( type, producer );
    _clear_id_bool[pid] = false;
  }

  /**
   * specify a data type and producername to be read.
   *
   * if at least one defined, then these are only the
   * data products read from the file. rest ignored.
   * if this never set, then by default, we read all
   * trees from the file
   *
   * @param[in] type ProductType_t number
   * @param[in] string Name of producer (in otherwords, the tree name in the file)
   *
   * effect is to modify _read_only_type and _read_only_name
   */
  void IOManager::specify_data_read( const ProductType_t type,
                                     const std::string& producername ) {
    _read_only_name.push_back( producername );
    _read_only_type.push_back( type );
  }

  void IOManager::reverse_all_products() {
    _reverse_all_image2d_products = true;
    _reverse_all_roi_products = true;
    _reverse_all_pixel2d_products = true;
  }

  int IOManager::addto_storeonly_list( ProductType_t type, std::string producername ) {
    _store_only_name.push_back( producername );
    _store_only_type.push_back( type );
    return _store_only_name.size();
  }
  
  /**
   * mark the image2d tree to be reversed
   *
   * @param[in] string name of image2d tree to be reversed
   *
   */
  void IOManager::reverse_pixel2d_tree( std::string name ) {
    _reverse_pixel2d_products.push_back( name );
  }

  /**
   * mark the roi tree to be reversed
   *
   * @param[in] string name of roi tree to be reversed
   *
   */
  void IOManager::reverse_roi_tree( std::string name ) {
    _reverse_roi_products.push_back( name );
  }
  
  
}
#endif
