#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "Processor/ProcessDriver.h"
#include "ReadJarrettFile.h"
#include "Run3DTracker.h"

int main(int nargs, char** argv){

    if(nargs < 3){
        std::cout << "usage : ./run_reco [cfg file] [input 1] [input 2] ..." << std::endl;
        return 1;
    }

    std::cout << "***********************" << std::endl;
    std::cout << "  RUN 3D RECO : START  " << std::endl;
    std::cout << "***********************" << std::endl;

    std::string cfg = argv[1];
    std::vector<std::string> data_inputs;
    for(int i = 2; i < nargs; i++){data_inputs.push_back(argv[i]);}

    larcv::ProcessDriver proc("ProcessDriver");
    proc.configure(cfg);
    proc.override_input_file(data_inputs);
    
    larcv::ProcessID_t algo_id = proc.process_id("Run3DTracker");
    larcv::Run3DTracker *algo = (larcv::Run3DTracker*)proc.process_ptr(algo_id);
    algo->SetSplineLocation("/Users/hourlier/Documents/PostDocMIT/Research/MicroBooNE/dllee_unified/LArCV/app/Reco3D/Proton_Muon_Range_dEdx_LAr_TSplines.root");
    algo->SetLLOutName("larlite_reco3D.root");
    std::cout << "spline and larlite out file should be set" << std::endl;
    proc.initialize();
    std::cout << "proc.initialize() done" << std::endl;
    proc.batch_process(0);
    std::cout << "RUN 3D RECO : about to finalize" << std::endl;
    proc.finalize();

    std::cout << "**********************" << std::endl;
    std::cout << "  RUN 3D RECO : STOP  " << std::endl;
    std::cout << "**********************" << std::endl;

    return 0;

}
