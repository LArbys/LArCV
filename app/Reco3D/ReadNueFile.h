#ifndef __READNUEFILE_H__
#define __READNUEFILE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "AStarTracker.h"

namespace larcv {

  class ReadNueFile : public ProcessBase {

  public:

    ReadNueFile(const std::string name="ReadNueFile");
    ~ReadNueFile(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

  private :
    int iTrack;
    larcv::AStarTracker tracker;
    TH2D *hEcomp;
    TH1D *hAverageIonization;
    TH1D *hEcomp1D;
    TH1D *hEcomp1D_m;
    TH1D *hEcomp1D_p;
    TH2D *hEcompdQdx;
    TH2D *hIonvsLength;
    double Ep_t;
    double Em_t;
    int run;
    int subrun;
    int event;
    std::vector<std::string> checkEvents;
    void MCevaluation();
    

  };

  class ReadNueFileProcessFactory : public ProcessFactoryBase {
  public:
    ReadNueFileProcessFactory() { ProcessFactory::get().add_factory("ReadNueFile",this); }
    ~ReadNueFileProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new ReadNueFile(instance_name); }
  };
    
}

#endif

