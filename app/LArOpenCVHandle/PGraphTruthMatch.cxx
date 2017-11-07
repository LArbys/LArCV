#ifndef __PGRAPHTRUTHMATCH_CXX__
#define __PGRAPHTRUTHMATCH_CXX__

#include "PGraphTruthMatch.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"
#include "DataFormat/EventROI.h"

#include "LArbysUtils.h"

namespace larcv {

  static PGraphTruthMatchProcessFactory __global_PGraphTruthMatchProcessFactory__;

  PGraphTruthMatch::PGraphTruthMatch(const std::string name)
    : ProcessBase(name)
  {}
    
  void PGraphTruthMatch::configure(const PSet& cfg)
  {
    LARCV_INFO() << "start" << std::endl;
    _adc_img2d_prod     = cfg.get<std::string>("ADCImageProducer");
    _true_img2d_prod    = cfg.get<std::string>("TrueImageProducer");
    _reco_pgraph_prod   = cfg.get<std::string>("RecoPGraphProducer");
    _reco_pixel_prod    = cfg.get<std::string>("RecoPxProducer");
    LARCV_INFO() << "end" << std::endl;
  }
  
  void PGraphTruthMatch::initialize()
  {
    LARCV_DEBUG() << "start" << std::endl;
    _tree = new TTree("PGraphTruthMatch","");
    _tree->Branch("run"    , &_run    , "run/I");    
    _tree->Branch("subrun" , &_subrun , "subrun/I");    
    _tree->Branch("event"  , &_event  , "event/I");    
    _tree->Branch("entry"  , &_entry  , "entry/I");    

    _tree->Branch("vtxid"       , &_vtxid, "vtxid/I");
    _tree->Branch("vtx_x"       , &_vtx_x, "vtx_x/F");
    _tree->Branch("vtx_y"       , &_vtx_y, "vtx_y/F");
    _tree->Branch("vtx_z"       , &_vtx_z, "vtx_z/F");
    _tree->Branch("vtx_on_nu"   , &_vtx_on_nu, "vtx_on_nu/I");
    _tree->Branch("vtx_on_nu_v" , &_vtx_on_nu_v);


    _tree->Branch("unknownfrac_v" , &_unknownfrac_v);
    _tree->Branch("electronfrac_v", &_electronfrac_v);
    _tree->Branch("gammafrac_v"   , &_gammafrac_v);
    _tree->Branch("pizerofrac_v"  , &_pizerofrac_v);
    _tree->Branch("muonfrac_v"    , &_muonfrac_v);
    _tree->Branch("kminusfrac_v"  , &_kminusfrac_v);
    _tree->Branch("piminusfrac_v" , &_piminusfrac_v);
    _tree->Branch("protonfrac_v"  , &_protonfrac_v);

    _tree->Branch("par_npx_v", &_par_npx_v);
    _tree->Branch("par_id_v" , &_par_id_v);
    _tree->Branch("par_id_vv", &_par_id_vv);

    LARCV_DEBUG() << "end" << std::endl;
  }
  
  bool PGraphTruthMatch::process(IOManager& mgr)
  {
    ClearVertex();
    auto ev_adc_img     = (EventImage2D*) mgr.get_data(kProductImage2D , _adc_img2d_prod);

    _run    = (int)ev_adc_img->run();
    _subrun = (int)ev_adc_img->subrun();
    _event  = (int)ev_adc_img->event();
    _entry  = (int)mgr.current_entry();

    if (_true_img2d_prod.empty()) {
      _tree->Fill();
      return true;
    }
    
    auto ev_seg_img     = (EventImage2D*) mgr.get_data(kProductImage2D , _true_img2d_prod);
    auto ev_reco_pgraph = (EventPGraph*)  mgr.get_data(kProductPGraph  , _reco_pgraph_prod);
    auto ev_reco_pix    = (EventPixel2D*) mgr.get_data(kProductPixel2D , _reco_pixel_prod);

    const auto& adc_img_v   = ev_adc_img->Image2DArray();
    const auto& seg_img_v   = ev_seg_img->Image2DArray();
    const auto& pgraph_v    = ev_reco_pgraph->PGraphArray();
    const auto& pix_m       = ev_reco_pix->Pixel2DClusterArray();
    const auto& pix_meta_m  = ev_reco_pix->ClusterMetaArray();

    LARCV_DEBUG() << "GOT: " << pgraph_v.size() << " vertices" << std::endl;
    for(size_t pgraph_id = 0; pgraph_id < pgraph_v.size(); ++pgraph_id) {    
      ClearVertex();

      _vtxid = (int)pgraph_id;

      LARCV_DEBUG() << "@pgraph_id=" << pgraph_id << std::endl;

      auto const& pgraph        = pgraph_v.at(pgraph_id);
      auto const& roi_v         = pgraph.ParticleArray();
      auto const& cluster_idx_v = pgraph.ClusterIndexArray();

      //
      // vertex on NU pixels ?
      //
      const auto reco_x = roi_v.front().X();
      const auto reco_y = roi_v.front().Y();
      const auto reco_z = roi_v.front().Z();

      _vtx_x = reco_x;
      _vtx_y = reco_y;
      _vtx_z = reco_z;

      _vtx_on_nu_v.resize(3,0);
      for(size_t plane=0; plane<3; ++plane) {

	const auto& plane_img = seg_img_v.at(plane);
	const auto& pmeta = plane_img.meta();
	const int nrows = plane_img.meta().rows();
	const int ncols = plane_img.meta().cols();

	double xpixel = larcv::kINVALID_DOUBLE;
	double ypixel = larcv::kINVALID_DOUBLE;
	Project3D(pmeta,
		  reco_x,reco_y,reco_z,0.0,
		  plane,
		  xpixel, ypixel);

	int xx = (int) (xpixel+0.5);
	int yy = (int) (ypixel+0.5);

	yy = nrows - yy - 1;

	float pixel_type = 0.0;
	pixel_type += plane_img.pixel(yy,xx);

	// check a ``box" around this projected location
	int yp1 = yy+1;
	int xp1 = xx+1;

	int ym1 = yy-1;
	int xm1 = xx-1;

	// (yp,x) (yp,xp) (yp,xm)
	if (yp1 >= 0 and yp1 < nrows) {
	  pixel_type += plane_img.pixel(yp1,xx);
	  if (xp1 >= 0 and xp1 < ncols) pixel_type += plane_img.pixel(yp1,xp1);
	  if (xm1 >= 0 and xm1 < ncols) pixel_type += plane_img.pixel(yp1,xm1);

	}

	// (ym,x) (ym,xp) (ym,xm)
	if (ym1 >= 0 and ym1 < nrows) {
	  pixel_type += plane_img.pixel(ym1,xx);
	  if (xp1 >= 0 and xp1 < ncols) pixel_type += plane_img.pixel(ym1,xp1);
	  if (xm1 >= 0 and xm1 < ncols) pixel_type += plane_img.pixel(ym1,xm1);
	}
	
	// (y,xp) (y,xm)
	if(xp1>=0 and xp1<ncols) pixel_type += plane_img.pixel(yy,xp1);
	if(xm1>=0 and xm1<ncols) pixel_type += plane_img.pixel(yy,xm1);


	if (pixel_type > 0) _vtx_on_nu_v[plane] = 1;
      }
      
      _vtx_on_nu = 0;
      for(auto on_nu : _vtx_on_nu_v) {
	if (on_nu>0) _vtx_on_nu += 1;
	
      }

      //
      // particle enclose which NU daughters?
      //

      auto npars = roi_v.size();
      _unknownfrac_v.resize(npars);
      _electronfrac_v.resize(npars);
      _gammafrac_v.resize(npars);
      _pizerofrac_v.resize(npars);
      _muonfrac_v.resize(npars);
      _kminusfrac_v.resize(npars);
      _piminusfrac_v.resize(npars);
      _protonfrac_v.resize(npars);

      _par_id_v.resize(npars);
      _par_id_vv.resize(npars);
      _par_npx_v.resize(npars,0);
      LARCV_DEBUG() << "GOT: " << npars << " particles" << std::endl;

      for(size_t roid=0; roid < npars; ++roid) {
	
	std::vector<int> pixel_type_v(((int)larcv::kROITypeMax)+1,0);

	const auto& roi = roi_v[roid];
	auto cidx = cluster_idx_v.at(roid);
	
	for(size_t plane=0; plane<3; ++plane) {
	  
	  auto iter_pix = pix_m.find(plane);
	  if(iter_pix == pix_m.end())
	    continue;

	  auto iter_pix_meta = pix_meta_m.find(plane);
	  if(iter_pix_meta == pix_meta_m.end())
	    continue;
	  
	  auto const& pix_v      = (*iter_pix).second;
	  auto const& pix_meta_v = (*iter_pix_meta).second;
	  
	  auto const& pix      = pix_v.at(cidx);
	  auto const& pix_meta = pix_meta_v.at(cidx);
	  
	  const auto& plane_img = seg_img_v.at(plane);

	  for(const auto& px : pix) {
	    auto posx = pix_meta.pos_x(px.Y());
	    auto posy = pix_meta.pos_y(px.X());
	    auto row  = plane_img.meta().row(posy);
	    auto col  = plane_img.meta().col(posx);
	    
	    float pixel_type = (size_t) plane_img.pixel(row,col);
	    pixel_type_v.at((size_t)pixel_type) += 1;
	  }
	} // end plane

	
      int npixels = 0;
      for(auto px_count : pixel_type_v) 
	npixels += px_count;

      _par_npx_v[roid] = npixels;

      auto res_iter = std::max_element(std::begin(pixel_type_v), std::end(pixel_type_v));
      auto res_loc  = std::distance(std::begin(pixel_type_v), res_iter);
      _par_id_v[roid] = res_loc;

      auto& unknownfrac  = _unknownfrac_v[roid];
      auto& electronfrac = _electronfrac_v[roid];
      auto& gammafrac    = _gammafrac_v[roid];
      auto& pizerofrac   = _pizerofrac_v[roid];
      auto& muonfrac     = _muonfrac_v[roid];
      auto& kminusfrac   = _kminusfrac_v[roid];
      auto& piminusfrac  = _piminusfrac_v[roid];
      auto& protonfrac   = _protonfrac_v[roid];

      unknownfrac  = 0.0;
      electronfrac = 0.0;
      gammafrac    = 0.0;
      pizerofrac   = 0.0;
      muonfrac     = 0.0;
      kminusfrac   = 0.0;
      piminusfrac  = 0.0;
      protonfrac   = 0.0;
      
      if (npixels) {
	unknownfrac  = (float) pixel_type_v.at(0) / (float) npixels;
	electronfrac = (float) pixel_type_v.at(3) / (float) npixels;
	gammafrac    = (float) pixel_type_v.at(4) / (float) npixels;
	pizerofrac   = (float) pixel_type_v.at(5) / (float) npixels;
	muonfrac     = (float) pixel_type_v.at(6) / (float) npixels;
	kminusfrac   = (float) pixel_type_v.at(7) / (float) npixels;
	piminusfrac  = (float) pixel_type_v.at(8) / (float) npixels;
	protonfrac   = (float) pixel_type_v.at(9) / (float) npixels;
      }

      _par_id_vv[roid] = std::move(pixel_type_v);

      } // end this particle
      LARCV_DEBUG() << "... next" << std::endl;
      _tree->Fill();
    } // end this vertex 
    LARCV_DEBUG() << "end" << std::endl;  
    return true;
  }

  void PGraphTruthMatch::finalize()
  {
    LARCV_DEBUG() << "start" << std::endl;  
    _tree->Write();
    LARCV_DEBUG() << "end" << std::endl;  
  }
  
  void PGraphTruthMatch::ClearVertex() {
    _vtx_x = -1.0*kINVALID_FLOAT;
    _vtx_y = -1.0*kINVALID_FLOAT;
    _vtx_z = -1.0*kINVALID_FLOAT;

    _vtxid     = -1.0*kINVALID_INT;
    _vtx_on_nu = -1.0*kINVALID_INT;

    _vtx_on_nu_v.clear();

    _par_id_v.clear();
    _par_id_vv.clear();
    _par_npx_v.clear();
    
    _unknownfrac_v.clear();  
    _electronfrac_v.clear(); 
    _gammafrac_v.clear();    
    _pizerofrac_v.clear();   
    _muonfrac_v.clear();     
    _kminusfrac_v.clear();   
    _piminusfrac_v.clear();  
    _protonfrac_v.clear();   

  }


}
#endif
