#include <iostream>
#include <fstream>

// json include in core
#include "json/nlohmann/json.hpp"

int main( int nargs, char** argv ) {

  std::cout << "json tester" << std::endl;

  //std::ifstream i("prod_test.cfg");
  std::ifstream i("prod_fullchain_ssnet_combined_newtag_base_c10_union_server.cfg" );
  nlohmann::json j;
  i >> j;
  
  std::cout << j.dump(2) << std::endl;
  
  return 0;
}
