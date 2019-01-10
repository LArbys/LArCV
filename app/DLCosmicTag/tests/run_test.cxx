#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ROOT
#include "TApplication.h"

// LArCV
#include "Processor/ProcessDriver.h"
#include "Base/LArCVBaseUtilFunc.h"
#include "Base/larcv_logger.h"

#include "DLCosmicTag/DLCosmicTagVertexReco.h"

int main( int nargs, char** argv ) {
  
  std::cout << "This script is used to test the DLCosmicVertexReco" << std::endl;
  
  std::string larcv_input   = argv[1];
  std::string cfg           = "prod_dlcosmictag_vertexreco.cfg";
  
  std::vector<std::string> larcv_input_v;
  larcv_input_v.push_back( larcv_input );

  // auto pset = larcv::CreatePSetFromFile(cfg);
  // auto proccfg = pset.get<larcv::PSet>( "ProcessDriver" );  
  // std::cout << proccfg.dump() << std::endl;

  larcv::ProcessDriver proc("ProcessDriver");  
  proc.configure( cfg );
  proc.override_input_file( larcv_input_v );
  proc.override_output_file( "output_test_dlcosmictagreco.root" );
  proc.override_output_file( "output_test_dlcosmictagreco.root" );  
  proc.initialize();

  int nentries = proc.io().get_n_entries();

  larcv::logger logger = larcv::logger::get_shared();

  std::stringstream msg;
  msg << "Number of events: " << nentries << std::endl;
  logger.send((larcv::msg::Level_t)2,"Setup") << msg.str();

  TApplication app( "app", &nargs, argv );
  
  for ( size_t ientry=0; ientry<nentries; ientry++ ) {
  //for ( size_t ientry=4; ientry<5; ientry++ ) {
    logger.send((larcv::msg::Level_t)2,"EVENT LOOP") << "entry " << ientry << std::endl;
    proc.process_entry(ientry);
  }
  
  logger.send((larcv::msg::Level_t)2,"POST") << "finalize" << std::endl;
  proc.finalize();

  logger.send((larcv::msg::Level_t)2,"POST") << "FIN" << std::endl;  

  return 0;
}
