#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name),
      _LArbysImageMaker(),
      _mc_chain(nullptr),
      _reco_chain(nullptr),
      _reco_vertex_v(nullptr),
      _particle_cluster_vvv(nullptr),
      _track_cluster_comp_vvv(nullptr)
  {}
  
  void LArbysImageAna::configure(const PSet& cfg)
  {
    _adc_producer = cfg.get<std::string>("ADCImageProducer");
    _LArbysImageMaker.Configure(cfg.get<larcv::PSet>("LArbysImageMaker"));
    _mc_tree_name = cfg.get<std::string>("MCTreeName");
    _reco_tree_name = cfg.get<std::string>("RecoTreeName");
  }

  void LArbysImageAna::initialize()
  {

    if (_input_larbys_root_file.empty()) throw larbys("No input root file specified");
    LARCV_DEBUG() << "Setting input ROOT file... " << _input_larbys_root_file << std::endl;
    _mc_chain = new TChain(_mc_tree_name.c_str());
    _mc_chain->AddFile(_input_larbys_root_file.c_str());
    _mc_chain->SetBranchAddress("run",&_mc_run);
    _mc_chain->SetBranchAddress("subrun",&_mc_subrun);
    _mc_chain->SetBranchAddress("event",&_mc_event);
    _mc_chain->SetBranchAddress("entry",&_mc_entry);
    _mc_index=0;
    _mc_chain->GetEntry(_mc_index);
    _mc_entries = _mc_chain->GetEntries();
    
    _reco_chain = new TChain(_reco_tree_name.c_str());
    _reco_chain->AddFile(_input_larbys_root_file.c_str());
    _reco_chain->SetBranchAddress("run",&_reco_run);
    _reco_chain->SetBranchAddress("subrun",&_reco_subrun);
    _reco_chain->SetBranchAddress("event",&_reco_event);
    _reco_chain->SetBranchAddress("entry",&_reco_entry);
    _reco_chain->SetBranchAddress("Vertex3D_v",&_reco_vertex_v);
    _reco_chain->SetBranchAddress("ParticleCluster_vvv",&_particle_cluster_vvv);
    //_reco_chain->SetBranchAddress("TrackClusterCompound_vvv",&_track_cluster_comp_vvv);
    _reco_index=0;
    _reco_chain->GetEntry(_reco_index);
    _reco_entries = _reco_chain->GetEntries();
  }

  bool LArbysImageAna::increment(uint entry)
  {

    _mc_chain->GetEntry(_mc_index);
    _reco_chain->GetEntry(_reco_index);

    if (_mc_entry<entry && _mc_index!=_mc_entries)
      { _mc_index++; }
    if (_reco_entry<entry && _reco_index!=_reco_entries)
      { _reco_index++; }

    if ( entry     != _reco_entry) return false;
    if ( entry     !=  _mc_entry ) return false;
    if ( _mc_entry != _reco_entry) return false;

    return true;

  }
  
  bool LArbysImageAna::process(IOManager& mgr)
  {
    uint entry = mgr.current_entry();

    LARCV_DEBUG() << "(this,mc,reco) entry & index @ (" << entry <<","<<_mc_entry<<","<<_reco_entry<<")"
		  << " & " << "(-,"<<_mc_index<<","<<_reco_index<<")"<<std::endl;

    if ( !increment(entry) ) return false;

    LARCV_INFO() << "(this,mc,reco) entry & index @ (" << entry <<","<<_mc_entry<<","<<_reco_entry<<")"
		 << " & " << "(-,"<<_mc_index<<","<<_reco_index<<")"<<std::endl;
    
    _adc_mat_v = _LArbysImageMaker.ExtractMat(mgr,_adc_producer);

    return true;
  }
  
  void LArbysImageAna::finalize()
  {}

}
#endif
