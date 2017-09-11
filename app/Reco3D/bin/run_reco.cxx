#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "Processor/ProcessDriver.h"

int main(int nargs, char** argv){

    if(nargs < 3){
        std::cout << "usage : ./run_reco [cfg file] [input 1] [input 2] ..." << std::endl;
        return 1;
    }

    std::cout << "***********************" << std::endl;
    std::cout << "  RUN 3D RECO : STRAT  " << std::endl;
    std::cout << "***********************" << std::endl;

    std::string cfg = argv[1];
    std::vector<std::string> data_inputs;
    for(int i = 2; i < nargs; i++){data_inputs.push_back(argv[i]);}

    larcv::ProcessDriver proc("ProcessDriver");
    proc.configure(cfg);
    proc.override_input_file(data_inputs);
    proc.initialize();
    proc.batch_process();
    proc.finalize();

    std::cout << "**********************" << std::endl;
    std::cout << "  RUN 3D RECO : STOP  " << std::endl;
    std::cout << "**********************" << std::endl;

    return 0;

}
