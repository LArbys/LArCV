#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// larlite
#include "DataFormat/storage_manager.h"
#include "APILArLite/Supera.h"
#include "Analysis/ana_processor.h"

int main( int nargs, char** argv ) {

  std::cout << "***********************" << std::endl;
  std::cout << " RUN SUPERA" << std::endl;
  std::cout << "***********************" << std::endl;

  if ( nargs<3 ) {
    std::cout << "usage: supera [cfg file] [output file] [input 1] [input 2] [input 3] [input 4]" << std::endl;
    return 0;
  }

  std::string cfg = argv[1];
  std::string out = argv[2];
  std::vector<std::string> inputlist;
  for (int i=3;i<nargs; i++) {
    inputlist.push_back( std::string(argv[i]) );
  }

  // check output does not exist. else stop in order to prevent mistaken overwrite
  std::ifstream f(out.c_str());
  if ( f.good() ) {
    std::cout << "output file, " << out << ", already exists. remove first." << std::endl;
    return 0;
  }
  
  // create ana_processor instance
  larlite::ana_processor ana;
  for (auto& fname : inputlist ) {
    ana.add_input_file( fname );
  }
  ana.set_io_mode( larlite::storage_manager::kREAD );
  ana.set_ana_output_file("");
  
  // create Supera
  larlite::Supera supera;
  supera.set_config( cfg );
  supera.supera_fname( out );

  // add supera to ana_processor
  ana.add_process( &supera );

  ana.run();

  std::cout << "Finished running ana_processor Supera event loop!" << std::endl;
  
  
  return 0;
}
