#ifndef __VERTEXFILTER_CXX__
#define __VERTEXFILTER_CXX__

#include "VertexFilter.h"

#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

#include <cassert>

namespace larcv {

  static VertexFilterProcessFactory __global_VertexFilterProcessFactory__;

  VertexFilter::VertexFilter(const std::string name)
    : ProcessBase(name), _tree(nullptr)
  {}
    
  void VertexFilter::configure(const PSet& cfg)
  {
    _in_pg_prod   = cfg.get<std::string>("InputPGraphProducer");
    _in_ctor_prod = cfg.get<std::string>("InputCtorProducer","");
    _in_img_prod  = cfg.get<std::string>("InputImgProducer","");

    _out_pg_prod   = cfg.get<std::string>("OutputPGraphProducer");
    _out_ctor_prod = cfg.get<std::string>("OutputCtorProducer","");
    _out_img_prod  = cfg.get<std::string>("OutputImgProducer","");
  }

  void VertexFilter::initialize() {
    
    _tree = new TTree("VertexFilterTree","");
    _tree->Branch("run"    , &_run    , "run/I");
    _tree->Branch("subrun" , &_subrun , "subrun/I");
    _tree->Branch("event"  , &_event  , "event/I");
    _tree->Branch("entry"  , &_entry  , "entry/I");
    _tree->Branch("cvtxid" , &_cvtxid , "cvtxid/I");
    _tree->Branch("fvtxid" , &_fvtxid , "fvtxid/I");
  }


  bool VertexFilter::process(IOManager& mgr)
  {
    LARCV_DEBUG() << "start" << std::endl;

    auto in_pg_v   = (EventPGraph*)  mgr.get_data(kProductPGraph,  _in_pg_prod);
    auto out_pg_v  = (EventPGraph*)  mgr.get_data(kProductPGraph,  _out_pg_prod);
    if (!(out_pg_v->PGraphArray().empty()))           throw larbys("data product not empty");

    EventPixel2D* in_ctor_v  = nullptr;
    EventPixel2D* in_img_v   = nullptr;
    EventPixel2D* out_ctor_v = nullptr;
    EventPixel2D* out_img_v  = nullptr;

    bool set_pixel = false;

    if (!_in_ctor_prod.empty() or !_in_img_prod.empty()) {
      if (_in_ctor_prod.empty() or _in_img_prod.empty()) {
	LARCV_CRITICAL() << "must specify both INPUT ctor and img producer" << std::endl;
	throw larbys();
      }

      in_ctor_v = (EventPixel2D*) mgr.get_data(kProductPixel2D, _in_ctor_prod);
      in_img_v  = (EventPixel2D*) mgr.get_data(kProductPixel2D, _in_img_prod);

      if (_out_ctor_prod.empty() or _out_img_prod.empty()) {
	LARCV_CRITICAL() << "must specify both OUTPUT ctor and img producer" << std::endl;
	throw larbys();
      }
      out_ctor_v = (EventPixel2D*) mgr.get_data(kProductPixel2D, _out_ctor_prod);
      out_img_v  = (EventPixel2D*) mgr.get_data(kProductPixel2D, _out_img_prod);
      
      if (!(out_ctor_v->Pixel2DClusterArray().empty())) throw larbys("data product not empty");
      if (!(out_img_v->Pixel2DClusterArray().empty()))  throw larbys("data product not empty");
      set_pixel = true;
    }


    _run    = (uint) in_pg_v->run();
    _subrun = (uint) in_pg_v->subrun();
    _event  = (uint) in_pg_v->event();
    _entry  = (uint) mgr.current_entry();
    
    //
    // nothing to do
    //
    if (_idx_v.empty()) return true;
    if (_par_v.empty() and set_pixel) return true;

    //
    // something to do
    //
    if (_idx_v.size() != in_pg_v->PGraphArray().size()) 
      throw larbys("pgraph & index vector size differ");
    
    int cluster_index_id = -1;
    for(size_t pgid=0; pgid < in_pg_v->PGraphArray().size(); ++pgid) {
      if (!_idx_v[pgid]) continue;
      LARCV_DEBUG() << "pass @pgid="<<pgid<<std::endl;
      const auto& pg_old = (*in_pg_v).PGraphArray()[pgid];

      //
      // don't filter the particles
      //
      if (!set_pixel) {
	out_pg_v->Append(pg_old);
	_cvtxid = pgid;
	_fvtxid = out_pg_v->PGraphArray().size() - 1;
	_tree->Fill();
	continue;
      }

      //
      // filter the particles and set type
      //

      const auto& par_pair = _par_v[pgid];
      if (par_pair.first  < 0) throw larbys("particle 0 invalid aho");
      if (par_pair.second < 0) throw larbys("particle 1 invalid aho");

      const auto& par_old_v = pg_old.ParticleArray();
      const auto& ci_old_v  = pg_old.ClusterIndexArray();

      assert (par_old_v.size() == 2);
      
      //
      // update particle label and store
      //
      auto roi0 = par_old_v.front();
      auto roi1 = par_old_v.back();

      roi0.Shape( par_pair.first  ? kShapeTrack : kShapeShower );
      roi1.Shape( par_pair.second ? kShapeTrack : kShapeShower );

      if (roi0.Shape() == kShapeShower) roi0.PdgCode(11);
      else roi0.PdgCode(14);

      if (roi1.Shape() == kShapeShower) roi1.PdgCode(11);
      else roi1.PdgCode(14);
      

      LARCV_DEBUG() << "set par0 shape=" << (int)roi0.Shape() << std::endl;
      LARCV_DEBUG() << "set par1 shape=" << (int)roi1.Shape() << std::endl;
      
      PGraph pg_new;

      cluster_index_id++;
      pg_new.Emplace(std::move(roi0),cluster_index_id);

      cluster_index_id++;
      pg_new.Emplace(std::move(roi1),cluster_index_id);

      out_pg_v->Emplace(std::move(pg_new));
      
      //
      // filter update particle ctor & imgx
      //
      auto const& cidx0 = ci_old_v.front();
      auto const& cidx1 = ci_old_v.back();

      LARCV_DEBUG() << "cidx0=" << cidx0 << std::endl;
      LARCV_DEBUG() << "cidx1=" << cidx1 << std::endl;

      auto const& ctor_m      = in_ctor_v->Pixel2DClusterArray();
      auto const& ctor_meta_m = in_ctor_v->ClusterMetaArray();
      
      LARCV_DEBUG() << "px ctor producer=" <<   _in_ctor_prod << std::endl;
      LARCV_DEBUG() << "ctor_m sz=     " << ctor_m.size() << std::endl;
      LARCV_DEBUG() << "ctor_meta_m sz=" << ctor_meta_m.size() << std::endl;
      
      auto const& pcluster_m      = in_img_v->Pixel2DClusterArray();
      auto const& pcluster_meta_m = in_img_v->ClusterMetaArray();

      LARCV_DEBUG() << "px img producer=" <<   _in_img_prod << std::endl;
      LARCV_DEBUG() << "pcluster_m sz=     " << pcluster_m.size() << std::endl;
      LARCV_DEBUG() << "pcluster_meta_m sz=" << pcluster_meta_m.size() << std::endl;

      for(size_t plane=0; plane<3; ++plane) {
	
	LARCV_DEBUG() << "@plane=" << plane << std::endl;

	auto iter_pcluster = pcluster_m.find(plane);
	if(iter_pcluster == pcluster_m.end()) {
	  LARCV_DEBUG() << "skip pcluster_m" << std::endl;
	  continue;
	}

	auto iter_pcluster_meta = pcluster_meta_m.find(plane);
	if(iter_pcluster_meta == pcluster_meta_m.end()) {
	  LARCV_DEBUG() << "skip pcluster_meta_m" << std::endl;
	  continue;
	}
	
	auto iter_ctor = ctor_m.find(plane);
	if(iter_ctor == ctor_m.end()) {
	  LARCV_DEBUG() << "skip ctor_m" << std::endl;
	  continue;
	}

	auto iter_ctor_meta = ctor_meta_m.find(plane);
	if(iter_ctor_meta == ctor_meta_m.end()) {
	  LARCV_DEBUG() << "skip ctor_meta_m" << std::endl;
	  continue;
	}

	LARCV_DEBUG() << "accepted!" << std::endl;

	auto const& pcluster_v      = (*iter_pcluster).second;
	auto const& pcluster_meta_v = (*iter_pcluster_meta).second;

	auto const& ctor_v      = (*iter_ctor).second;
	auto const& ctor_meta_v = (*iter_ctor_meta).second;

	//par0
	auto const& pcluster0      = pcluster_v.at(cidx0);
	auto const& pcluster_meta0 = pcluster_meta_v.at(cidx0);

	auto const& ctor0      = ctor_v.at(cidx0);
	auto const& ctor_meta0 = ctor_meta_v.at(cidx0);

	//par1
	auto const& pcluster1      = pcluster_v.at(cidx1);
	auto const& pcluster_meta1 = pcluster_meta_v.at(cidx1);

	auto const& ctor1      = ctor_v.at(cidx1);
	auto const& ctor_meta1 = ctor_meta_v.at(cidx1);

	out_ctor_v->Append(plane,ctor0,ctor_meta0);
	out_ctor_v->Append(plane,ctor1,ctor_meta1);

	out_img_v->Append(plane,pcluster0,pcluster_meta0);
	out_img_v->Append(plane,pcluster1,pcluster_meta1);
	
	LARCV_DEBUG() << "end this plane" << std::endl;
      } // end plane
      LARCV_DEBUG() << "end this pgraph" << std::endl;
      
      assert (!out_pg_v->PGraphArray().empty());

      _cvtxid = pgid;
      _fvtxid = out_pg_v->PGraphArray().size() - 1;
      _tree->Fill();
    } // end pgraph
    LARCV_DEBUG() << "end" << std::endl;
    clear();
    return true;
  }
  
  void VertexFilter::finalize() {
    assert(_tree);
    _tree->Write();
  }

  void VertexFilter::clear() {
    LARCV_DEBUG() << "start" << std::endl;
    _idx_v.clear();
    _par_v.clear();

    _run = kINVALID_INT;
    _subrun = kINVALID_INT;
    _event = kINVALID_INT;
    _entry = kINVALID_INT;

    _cvtxid = kINVALID_INT;
    _fvtxid = kINVALID_INT;

    LARCV_DEBUG() << "end" << std::endl;
  }

}
#endif
