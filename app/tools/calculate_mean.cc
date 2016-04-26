
#include "Base/larbys.h"
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include <map>
static larcv::logger& logger()
{ 
  static larcv::logger __main_logger__("main");
  return __main_logger__;
}

static void set_verbosity(larcv::msg::Level_t l)
{ logger().set(l); }

int main(int argc, char** argv){

  set_verbosity(larcv::msg::kNORMAL);

  if(argc < 3)  {
    LARCV_CRITICAL() << "Invalid argument count (needs at least 3)" << std::endl
		     << "  Usage: " << argv[0] << " PRODUCER_NAME FILE1 [FILE2 FILE3 ...]" << std::endl;
    return 1;
  }

  std::string producer(argv[1]);

  larcv::IOManager input_strm(larcv::IOManager::kREAD);
  for(int i=2; i<argc; ++i) input_strm.add_in_file(argv[i]);
  input_strm.initialize();

  larcv::IOManager output_strm(larcv::IOManager::kWRITE);
  output_strm.set_out_file("mean.root");
  output_strm.initialize();
  auto output_image = (larcv::EventImage2D*)(output_strm.get_data(larcv::kProductImage2D,"mean"));
  std::map<larcv::PlaneID_t,std::vector<double> > image_m;
  std::map<larcv::PlaneID_t,larcv::ImageMeta > meta_m;

  const size_t NPLANES=3;
  size_t nentries = input_strm.get_n_entries();
  size_t entries_fraction = nentries/10;
  size_t entry=0;
  double image_count=0;
  while(entry < nentries) {
    input_strm.read_entry(entry);
    ++entry;

    if(entry%entries_fraction==0)

      LARCV_NORMAL() << "Finished " << entry/entries_fraction * 10 << " %" << std::endl;

    auto event_image = (larcv::EventImage2D*)(input_strm.get_data(larcv::kProductImage2D,producer));

    if(event_image->Image2DArray().size()!=NPLANES) continue;
    
    if(image_m.empty()) {
      
      for(auto const& img : event_image->Image2DArray()) {

	auto const& vec = img.as_vector();
	auto plane = img.meta().plane();
	meta_m[plane] = img.meta();
	std::vector<double> vec_copy(vec.size(),0.);
	for(size_t i=0; i<vec.size(); ++i) vec_copy[i] = vec[i];
	image_m.emplace(plane,std::move(vec_copy));
      }
      
      if(image_m.size() != NPLANES) {
	LARCV_CRITICAL() << "Duplicate plane? # plane-image does not make sense..." << std::endl;
	throw larcv::larbys();
      }

    }else{

      for(auto const& img : event_image->Image2DArray()) {

	auto iter = image_m.find(img.meta().plane());
	if(iter == image_m.end()) {
	  LARCV_CRITICAL() << "PlaneID_t set changed during the event loop... unexpected..." << std::endl;
	  throw larcv::larbys();
	}
	auto& vec_copy = (*iter).second;
	auto const& vec = img.as_vector();
	if(vec.size() != vec_copy.size()) {
	  LARCV_CRITICAL() << "Image size has changed for plane " << (*iter).first << std::endl;
	  throw larcv::larbys();
	}
	
	for(size_t i=0; i<vec.size(); ++i) vec_copy[i] += vec[i];
      }
    }
    ++image_count;
  }

  for(auto& plane_img : image_m) {

    auto const& plane = plane_img.first;
    auto& vec_copy = plane_img.second;
    for(auto& v : vec_copy) v /= image_count;
    
    std::vector<float> vec(vec_copy.size(),0);
    for(size_t i=0; i<vec.size(); ++i) vec[i] = (float)(vec_copy[i]);

    larcv::ImageMeta meta(meta_m[plane]);
    larcv::Image2D img(std::move(meta),std::move(vec));
    output_image->Emplace(std::move(img));
  }
  output_strm.set_id(0,0,0);
  output_strm.save_entry();
  output_strm.finalize();
  input_strm.finalize();

  return 0;
}
