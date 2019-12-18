#include "ShowerMask.h"

namespace larcv {
namespace showermask{

void ShowerMask::configure( const larcv::PSet& pset ) {
    // all inputs needed to mask
    _input_wire_producer          = pset.get<std::string>("InputWireProducer");
    _input_ssnet_uplane_producer = pset.get<std::string>("InputSSNetUProducer");
    _input_ssnet_vplane_producer = pset.get<std::string>("InputSSNetVProducer");
    _input_ssnet_yplane_producer = pset.get<std::string>("InputSSNetYProducer");
    _input_vtx_producer   = pset.get<std::string>("InputVtxProducer");
    _input_hits_producer   = pset.get<std::string>("InputHitsProducer");

  }

  void ShowerMask::initialize() {
    OutFile = new TFile("ShowerMaskTest.root","RECREATE");
  }

bool ShowerMask::process( larcv::IOManager& io,larlite::storage_manager& ioll,
    larcv::IOManager& ioforward,larlite::storage_manager& outputdata){
  //set utils object
  Utils Utils;
  // inputs
  std::cout << "process"<<std::endl;
  OutFile->cd();

  const auto ssnet_uplane      = (larcv::EventImage2D*)ioforward.get_data( larcv::kProductImage2D, _input_ssnet_uplane_producer );
  const auto ssnet_vplane      = (larcv::EventImage2D*)ioforward.get_data( larcv::kProductImage2D, _input_ssnet_vplane_producer );
  const auto ssnet_yplane      = (larcv::EventImage2D*)ioforward.get_data( larcv::kProductImage2D, _input_ssnet_yplane_producer );
  const auto ev_img            = (larcv::EventImage2D*)io.get_data( larcv::kProductImage2D, _input_wire_producer );
  const auto ev_pgraph         = (larcv::EventPGraph*) io.get_data( larcv::kProductPGraph, _input_vtx_producer);
  const auto ev_hitsreco2d     = (larlite::event_hit*)ioll.get_data(larlite::data::kHit, _input_hits_producer);


  run    = ioll.run_id();
  subrun = ioll.subrun_id();
  event  = ioll.event_id();
  std::cout<<"run: "<<run<<" subrun: "<<subrun<<" event: "<<event<<std::endl;

  std::cout<<"number of hits: "<<ev_hitsreco2d->size()<<std::endl;

  //grab array wire image
  auto const& wire_img = ev_img->Image2DArray();
  auto const& wireu_meta = wire_img.at(0).meta();
  auto const& wirev_meta = wire_img.at(1).meta();
  auto const& wirey_meta = wire_img.at(2).meta();
  auto const& ssnetu_img = ssnet_uplane->Image2DArray();
  auto const& ssnetv_img = ssnet_vplane->Image2DArray();
  auto const& ssnety_img = ssnet_yplane->Image2DArray();
  auto const& ssnet_meta = ssnety_img.at(0).meta();

  //get reco vtx location
  //first get 3d version
  reco_vtx_location_3D_v = GetRecoVtxLocs(ev_pgraph);
  //get 2d location Projection
  for (int ii = 0; ii < reco_vtx_location_3D_v.size(); ii++){
    reco_vtx_location_2D_v.push_back(Utils.getProjectedPixel(reco_vtx_location_3D_v[ii], wirey_meta, 3));
  }
  std::cout<<"number of vertices: "<<reco_vtx_location_3D_v.size()<<std::endl;

  HitLocation( ev_hitsreco2d, wireu_meta, wirev_meta, wirey_meta);

  HitDistanceFromVtx();

  ChooseHitstoKeep(ssnetu_img,ssnetv_img,ssnety_img,.5);

  int totalsaved=0;
  for (int hit = 0;hit<hitstokeep_v.size();hit++){
     totalsaved+=hitstokeep_v.at(hit);
  }
  std::cout<<"number of hits kept: "<<totalsaved<<std::endl;
  
  //save to output file
  outputdata.set_id(run,subrun,event);
  larlite::event_hit* const ev_hits_out = (larlite::event_hit* )(outputdata.get_data(larlite::data::kHit, "maskedhits"));

  for (int hit =0;hit<ev_hitsreco2d->size();hit++){
    int keephit = hitstokeep_v.at(hit);
    if (keephit==1) ev_hits_out->push_back(ev_hitsreco2d->at(hit));
  }//end of loop through hits
  
  std::cout<<"size of output tree: "<<ev_hits_out->size()<<std::endl;
  reco_vtx_location_2D_v.clear();
  reco_vtx_location_3D_v.clear();
  hitdistances_v.clear();
  hit_location.clear();
  hitstokeep_v.clear();
  kepthits_v.clear();

  return true;
}

//------------------------------------------------------------------------------

std::vector<std::vector<double>> ShowerMask::GetRecoVtxLocs(larcv::EventPGraph* ev_pgraph){
  //get 3d locations of vtx points and save them all
  std::vector<std::vector<double>> all_vtx_locations;
  for(size_t pgraph_id = 0; pgraph_id < ev_pgraph->PGraphArray().size(); ++pgraph_id) {
    auto const& pgraph = ev_pgraph->PGraphArray().at(pgraph_id);
    std::vector<double> _vtx(3,0);
    _vtx[0] = pgraph.ParticleArray().front().X();
    _vtx[1] = pgraph.ParticleArray().front().Y();
    _vtx[2] = pgraph.ParticleArray().front().Z();
    all_vtx_locations.push_back(_vtx);
  }
  return all_vtx_locations;
}//end of function

//------------------------------------------------------------------------------
void ShowerMask::HitLocation(larlite::event_hit* ev_hitsreco2d,larcv::ImageMeta wireu_meta,
    larcv::ImageMeta wirev_meta,larcv::ImageMeta wirey_meta){
  //function to loop through hits and determine distance from closest vtx
  //loop through hits
  for(int hit =0;hit<ev_hitsreco2d->size();hit++){
    std::vector<float> HitLocation(3,-1.0);
    int plane = ev_hitsreco2d->at(hit).View();
    float row = -1;
    float col = -1;
    // std::cout<<hit<<","<<plane<<" ";

    if (plane == 0){
      if(ev_hitsreco2d->at(hit).PeakTime()+2401<8448){
        row = wireu_meta.row(ev_hitsreco2d->at(hit).PeakTime()+2401);
        col = wireu_meta.col(ev_hitsreco2d->at(hit).Channel());
      }
    }
    else if(plane == 1){
      if(ev_hitsreco2d->at(hit).PeakTime()+2401<8448){
        row = wirev_meta.row(ev_hitsreco2d->at(hit).PeakTime()+2401);
        col = wirev_meta.col(ev_hitsreco2d->at(hit).Channel()-2400);
      }
    }
    else{
      if(ev_hitsreco2d->at(hit).PeakTime()+2401<8448){
        row = wirey_meta.row(ev_hitsreco2d->at(hit).PeakTime()+2401);
        col = wirey_meta.col(ev_hitsreco2d->at(hit).Channel()-2*2400);
      }
    }

    HitLocation.at(0) = (float) plane;
    HitLocation.at(1) = row;
    HitLocation.at(2) = col;
    hit_location.push_back(HitLocation);
  }//end of hit LOOP
  return;
}
//------------------------------------------------------------------------------
void ShowerMask::HitDistanceFromVtx(){
  //function to loop through hits and determine distance from closest vtx
  //loop through hits

  for(int hit =0;hit<hit_location.size();hit++){
    float distance = 20000;
    if (hit_location.at(hit).at(1)>-1){
      int plane = hit_location.at(hit).at(0);
      float row = hit_location.at(hit).at(1);
      float col = hit_location.at(hit).at(2);
      for (int vtx =0;vtx<reco_vtx_location_2D_v.size();vtx++){
        float vtx_row = reco_vtx_location_2D_v.at(vtx).at(0);
        float vtx_col;
        if (plane == 0) vtx_col = reco_vtx_location_2D_v.at(vtx).at(1);
        else if (plane == 1) vtx_col = reco_vtx_location_2D_v.at(vtx).at(2);
        else vtx_col = reco_vtx_location_2D_v.at(vtx).at(3);
        float vtx_distance = std::sqrt((col-vtx_col)*(col-vtx_col)+(row-vtx_row)*(row-vtx_row));
        if (vtx_distance<distance)distance = vtx_distance;
      }//end of vtx loop
    }//end of hit in time window
    if (distance>5) hitdistances_v.push_back(0);
    else hitdistances_v.push_back(1);
  }//end of hit LOOP
  return;
}
//------------------------------------------------------------------------------
void ShowerMask::ChooseHitstoKeep(std::vector<larcv::Image2D> shower_score_u, std::vector<larcv::Image2D> shower_score_v,
                      std::vector<larcv::Image2D> shower_score_y, float threshold){
  /*
  This function takes in a vector of vector of ints, where the top vec is
  of hits, where each hit (inner vec) has 3 ints, row, column, and plane
  for the hit. It returns a vector of ints where you have 0 if the hit has
  sccore < threshold and 1 if the score is >= threshold. Threshold has default
  value of 0.5 if the arg is not given.
  It only does this is close_to_vertex_v[idx] is not 1
  */
  int num_hits = hit_location.size();
  int num_hits2 = hitdistances_v.size();
  assert (num_hits == num_hits2);
  for (int idx = 0; idx < num_hits; idx++){
    int hit_val = 0;
    int row = hit_location[idx][1];
    int col = hit_location[idx][2];
    int plane = hit_location[idx][0];
    float score = 0;
    // check shower image score for correct plane
    if (row != -1){
      if (plane == 0){score = shower_score_u.at(0).pixel(row,col);}
      else if (plane == 1){score = shower_score_v.at(0).pixel(row,col);}
      else{score = shower_score_y.at(0).pixel(row,col);}
    }

    if (score >= threshold ||  hitdistances_v[idx] == 1 )hitstokeep_v.push_back(1);
    else hitstokeep_v.push_back(0);
  }//end of loop over hits
  return;
}
//------------------------------------------------------------------------------

void ShowerMask::finalize() {

  OutFile->cd();
  OutFile->Write();

}
}//end of showermask namespace
}//end of ublarcvapp namespace
