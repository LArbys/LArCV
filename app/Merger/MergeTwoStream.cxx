#ifndef MERGETWOSTREAM_CXX
#define MERGETWOSTREAM_CXX

#include "MergeTwoStream.h"
#include "Base/LArCVBaseUtilFunc.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventChStatus.h"

namespace larcv {

  MergeTwoStream::MergeTwoStream() : larcv_base        ( "MergeTwoStream"   )
				   , _nu_driver        ( "NeutrinoMCStream" )
				   , _nu_proc          ( nullptr            )
				   , _nu_proc_name     ( ""                 )
				   , _cosmic_driver    ( "CosmicDataStream" )
				   , _cosmic_proc      ( nullptr            )
				   , _cosmic_proc_name ( ""                 )
				   , _merge_driver     ( "OutStream"        )
				   , _prepared         (false)
				   , _num_nu           (0)
				   , _num_cosmic       (0)
				   , _num_max          (0)
				   , _num_frac         (0)
  {}

  void MergeTwoStream::override_input_file(const std::vector<std::string>& nu_flist,
					   const std::vector<std::string>& cosmic_flist)
  {
    if(_prepared) {
      LARCV_CRITICAL() << "Cannot re-configure after initialized..." << std::endl;
      throw larbys();
    }
    if(nu_flist.size())
      _nu_driver.override_input_file(nu_flist);
    if(cosmic_flist.size())
      _cosmic_driver.override_input_file(cosmic_flist);
  }

  void MergeTwoStream::override_output_file(std::string out_fname)
  {
    if(_prepared) {
      LARCV_CRITICAL() << "Cannot re-configure after initialized..." << std::endl;
      throw larbys();
    }
    _io.set_out_file(out_fname);
  }

  void MergeTwoStream::configure(std::string cfg_file)
  {
    if(_prepared) {
      LARCV_CRITICAL() << "Cannot re-configure after initialized..." << std::endl;
      throw larbys();
    }
    auto main_cfg = CreatePSetFromFile(cfg_file);
    auto const& cfg = main_cfg.get_pset(name());
    if(!cfg.contains_pset(_cosmic_driver.name())) {
      LARCV_CRITICAL() << "CosmicDataStream parameter set not found..." << std::endl;
      throw larbys();
    }
    if(!cfg.contains_pset(_nu_driver.name())) {
      LARCV_CRITICAL() << "NeutrinoMCStream parameter set not found..." << std::endl;
      throw larbys();
    }
    if(!cfg.contains_pset(_merge_driver.name())) {
      LARCV_CRITICAL() << "OutStream parameter set not found..." << std::endl;
      throw larbys();
    }
    _cosmic_proc_name = cfg.get<std::string>("CosmicImageHolder");
    if(_cosmic_proc_name.empty()) {
      LARCV_CRITICAL() << "CosmicImageHolder name is empty" << std::endl;
      throw larbys();
    }
    _nu_proc_name = cfg.get<std::string>("NeutrinoImageHolder");
    if(_nu_proc_name.empty()) {
      LARCV_CRITICAL() << "NeutrinoImageHolder name is empty" << std::endl;
      throw larbys();
    }
    _merge_proc_name = cfg.get<std::string>("ImageMerger");
    if(_merge_proc_name.empty()) {
      LARCV_CRITICAL() << "ImageMerger name is empty" << std::endl;
      throw larbys();
    }

    LARCV_WARNING() << "Note CosmicDataStream should contain image to-be-overlayed with ROI (Stream2 is the base image)"<<std::endl;
    LARCV_NORMAL() << "Registered Input CosmicData: " << _cosmic_proc_name << std::endl;
    LARCV_NORMAL() << "Registered Input NeutrinoMC: " << _nu_proc_name << std::endl;
    LARCV_NORMAL() << "Registered Output Merged   : " << _merge_proc_name << std::endl; 

    set_verbosity((msg::Level_t)(cfg.get<unsigned short>("Verbosity",logger().level())));
    
    _merge_driver.configure(cfg.get_pset(_merge_driver.name()));
    _cosmic_driver.configure(cfg.get_pset(_cosmic_driver.name()));
    _nu_driver.configure(cfg.get_pset(_nu_driver.name()));
  }
  
  void MergeTwoStream::initialize()
  {
    _merge_driver.initialize();
    _cosmic_driver.initialize();
    _nu_driver.initialize();
    // retrieve image holder 1 & 2
    auto const id_cosmic = _cosmic_driver.process_id(_cosmic_proc_name);
    _cosmic_proc = (ImageHolder*)(_cosmic_driver.process_ptr(id_cosmic));
    auto const id_nu = _nu_driver.process_id(_nu_proc_name);
    _nu_proc = (ImageHolder*)(_nu_driver.process_ptr(id_nu));
    auto const id_merged = _merge_driver.process_id(_merge_proc_name);
    _merge_proc = (ImageMerger*)(_merge_driver.process_ptr(id_merged));
    _merge_proc->NeutrinoImageHolder(_nu_proc);
    _merge_proc->CosmicImageHolder(_cosmic_proc);
    _prepared=true;
    _num_cosmic = 0;
    _num_nu = 0;
    _num_max = std::min(_cosmic_driver.io().get_n_entries(),_nu_driver.io().get_n_entries());
    _num_frac = _num_max/10 + 1;
    LARCV_NORMAL() << "Processing max " << _num_max << " entries..." << std::endl;
  }
  
  bool MergeTwoStream::process()
  {
    if(!_prepared) {
      LARCV_CRITICAL() << "Must call initialize() beore process()" << std::endl;
      throw larbys();
    }

    while(_num_cosmic < _num_max) {
      ++_num_cosmic;
      if(_cosmic_driver.process_entry()) break;
    }
    while(_num_nu < _num_max) {
      ++_num_nu;
      if(_nu_driver.process_entry()) break;
    }

    if(_num_cosmic >= _num_max) return false;
    if(_num_nu >= _num_max) return false;

    LARCV_INFO() << "Processing CosmicDataStream entry " << _num_cosmic
		 << " ... NeutrinoMCStream entry " << _num_nu << std::endl;

    static size_t num_entry=0;
    _merge_driver.process_entry();
    ++num_entry;
    if(_num_max < 10)
      { LARCV_NORMAL() << "Processed " << num_entry << " entries..." << std::endl; }
    else if( num_entry && (num_entry%_num_frac == 0) )
      { LARCV_NORMAL() << "Processed " << 10*((num_entry/_num_frac)+1) << " % processed..." << std::endl; }

    static size_t ctr=0;
    ctr++;
    return (ctr<3);
    //return true;
  }

  void MergeTwoStream::batch_process()
  {
    while(1) if(!process()) break;
    LARCV_NORMAL() << "Finished 100%" << std::endl;
  }
  
  void MergeTwoStream::finalize()
  {
    if(!_prepared) {
      LARCV_CRITICAL() << "Must call initialize() beore process()" << std::endl;
      throw larbys();
    }

    _nu_driver.finalize();
    _cosmic_driver.finalize();
    _merge_driver.finalize();
  }

}
#endif
