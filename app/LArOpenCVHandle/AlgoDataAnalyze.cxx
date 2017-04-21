#ifndef __ALGODATAANALYZE_CXX__
#define __ALGODATAANALYZE_CXX__

#include "AlgoDataAnalyze.h"

namespace larcv {

  static AlgoDataAnalyzeProcessFactory __global_AlgoDataAnalyzeProcessFactory__;

  AlgoDataAnalyze::AlgoDataAnalyze(const std::string name)
    : ProcessBase(name), _larocvserial(nullptr), _reco_chain(nullptr), _vertex_tree(nullptr)
  {}
  
  void AlgoDataAnalyze::configure(const PSet& cfg)
  {
    _input_larbys_reco_root_file = cfg.get<std::string>("InputLArbysImageFile");
    _reco_tree_name = cfg.get<std::string>("RecoTreeName");
    _serial_name    = cfg.get<std::string>("LArOCVSerialName");
  }

  void AlgoDataAnalyze::initialize()
  {
    _reco_chain = new TChain(_reco_tree_name.c_str());
    _reco_chain->AddFile(_input_larbys_reco_root_file.c_str());
    _reco_chain->SetBranchAddress("run",&_reco_run);
    _reco_chain->SetBranchAddress("subrun",&_reco_subrun);
    _reco_chain->SetBranchAddress("event",&_reco_event);
    _reco_chain->SetBranchAddress("entry",&_reco_entry);
    _reco_chain->SetBranchAddress(_serial_name.c_str(),&_larocvserial);
    _reco_index = 0;
    _reco_chain->GetEntry(_reco_index);
    _reco_entries = _reco_chain->GetEntries();
    
    _vertex_tree = new TTree("OutTree","OutTree");
    _vertex_tree->Branch("run"   ,&_reco_run,"run/i");
    _vertex_tree->Branch("subrun",&_reco_subrun,"subrun/i");
    _vertex_tree->Branch("event" ,&_reco_event,"event/i");
    _vertex_tree->Branch("entry" ,&_reco_entry,"entry/i");    
    _vertex_tree->Branch("x",&_reco_vertex_x,"x/D");
    _vertex_tree->Branch("y",&_reco_vertex_y,"y/D");
    _vertex_tree->Branch("z",&_reco_vertex_z,"z/D");

  }

  void AlgoDataAnalyze::finalize()
  {
    LARCV_INFO() << "See... " << _reco_entries << "... reco entries" << std::endl;
    for(size_t reco_entry=0; reco_entry<_reco_entries; ++reco_entry) {
      if (reco_entry)
	_reco_chain->GetEntry(reco_entry);

      if(!_larocvserial)
	throw larbys("Invalid pointer to LArOCVSerial type");
      
      LARCV_DEBUG() << reco_entry << ") @("
		    <<_reco_run    <<","
		    <<_reco_subrun <<","
		    <<_reco_event  <<","
		    <<_reco_entry  <<") vtx sz: "
		    << _larocvserial->_vertex_v.size() << std::endl;
      for(const auto& vtx : _larocvserial->_vertex_v) {
	_reco_vertex_x = vtx.x;
	_reco_vertex_y = vtx.y;
	_reco_vertex_z = vtx.z;
	_vertex_tree->Fill();
      }
      
    }
    ana_file().cd();
    _vertex_tree->Write();
  }

}
#endif
