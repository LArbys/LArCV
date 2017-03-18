#ifndef __LARBYSIMAGEPID_CXX__
#define __LARBYSIMAGEPID_CXX__

#include "LArbysImagePID.h"

namespace larcv {

  static LArbysImagePIDProcessFactory __global_LArbysImagePIDProcessFactory__;

  LArbysImagePID::LArbysImagePID(const std::string name, 
				 const std::vector<std::string> particle_types, 
				 const std::vector<std::string> interaction_types)
    : ProcessBase(name)
    , _particle_types(particle_types)
    , _interaction_types(interaction_types)
  {}
    
  void LArbysImagePID::Clear(){
    _ptype_ctr.clear();
    _ptype_ctr.resize(5,0);

    _rois_score.clear();
    _rois_score.resize(2);
    _ptype_scores.clear();
    
    //_roi_score.resize(5,0.0);
    //_rois_score[0]=_roi_score;
    //_rois_score[1]=_roi_score;

    
    _interaction_scores.clear();
    _interaction_scores.resize(2,0.0);
  }
  
  void LArbysImagePID::configure(const PSet& cfg)
  {
    _use_shape = cfg.get<bool>("UseShape");
  }

  void LArbysImagePID::initialize()
  {}

  bool LArbysImagePID::process(IOManager& mgr)
  {
    Clear();
    _ptype_ctr.resize(5,0);
    
    auto ev_pgraph = (EventPGraph*)mgr.get_data(kProductPGraph,"test");
    _run    = ev_pgraph->run();
    _subrun = ev_pgraph->subrun();
    _event  = ev_pgraph->event();
    _entry  = mgr.current_entry();
    
    auto pgraphs = ev_pgraph->PGraphArray();
    
    LARCV_DEBUG()<< "Entry: "<<_entry
		 <<", Run: "<<_run
		 <<", Subrun: "<<_subrun
		 <<", Event: "<<_event
		 <<", has "<<pgraphs.size()<<" vertices."
		 <<std::endl;

    _ptype_scores.resize(pgraphs.size());
    LARCV_DEBUG()<<"number of vtx is "<<pgraphs.size()<<std::endl;
    std::vector<std::vector<float>> ptype_scor(2, std::vector<float>(5,0.0));
    for(auto &ptype_score : _ptype_scores) ptype_score = ptype_scor;
        
    for (size_t vtx_idx=0; vtx_idx< pgraphs.size(); ++vtx_idx)
      {
	auto rois_obj = pgraphs[vtx_idx];
	if (rois_obj.NumParticles()!=2){ 
	  LARCV_CRITICAL()<<"    The vertex has >>>> "<< rois_obj.NumParticles()<<" <<<< rois !"<<std::endl;
	  //throw larbys("Go to match, LArbysImageFilter(LArbysImage)");
	}
	
	auto rois = rois_obj.ParticleArray();
	for (int roi_idx=0; roi_idx<2; ++roi_idx){
	  auto roi = rois[roi_idx];
	  _rois_score[roi_idx] = roi.TypeScore();
	  if (_use_shape){
	    if (roi.Shape()==0&& roi.Type()==0){
	      _ptype_ctr[roi.Type()]++;  
	      _ptype_scores[vtx_idx][roi_idx][roi.Type()]=roi.TypeScore()[roi.Type()];
	    }
	    if (roi.Shape()==0&& roi.Type()==1){
	      _ptype_ctr[roi.Type()]++;  
	      _ptype_scores[vtx_idx][roi_idx][roi.Type()]=roi.TypeScore()[roi.Type()];
	    }
	    if (roi.Shape()==1&& roi.Type()==2){
	      _ptype_ctr[roi.Type()]++;  
	      _ptype_scores[vtx_idx][roi_idx][roi.Type()]=roi.TypeScore()[roi.Type()];
	    }
	    if (roi.Shape()==0&& roi.Type()==3){
	      _ptype_ctr[roi.Type()]++;  
	      _ptype_scores[vtx_idx][roi_idx][roi.Type()]=roi.TypeScore()[roi.Type()];
	    }
	    if (roi.Shape()==1&& roi.Type()==4){
	      _ptype_ctr[roi.Type()]++;  
	      _ptype_scores[vtx_idx][roi_idx][roi.Type()]=roi.TypeScore()[roi.Type()];
	    }
	  }
	  if(!_use_shape){
	    for (int type=0; type<5; ++type){
	      if (roi.Type()==type) {
		_ptype_ctr[type]++;
		_ptype_scores[vtx_idx][roi_idx][type]=roi.TypeScore()[type];
	      }
	    }
	  }
	  //LARCV_DEBUG()<<roi.Index()<<std::endl;
	  //LARCV_DEBUG()<<particle_types[roi.Type()]<<std::endl;
	}
	
	for (auto x : _ptype_ctr) std::cout<< x <<" ";
	std::cout<<std::endl;
	for (auto scores : _rois_score) {
	  for (auto score : scores) std::cout<<score<<" ";
	  std::cout<<std::endl;
	}
	std::cout<<std::endl;
	for (auto scores : _ptype_scores) {
	  for (auto score : scores) {
	    for (auto scor : score)std::cout<<scor<<" ";
	  }
	  std::cout<<std::endl;
	}
	std::cout<<std::endl;
	_ptype_ctr.clear();
	_ptype_ctr.resize(5,0);
      }
  }
  
  void LArbysImagePID::finalize()
  {}

}
#endif
