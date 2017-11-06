#ifndef __PGRAPHTRUTHMATCH_H__
#define __PGRAPHTRUTHMATCH_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

namespace larcv {

  class PGraphTruthMatch : public ProcessBase {

  public:
    PGraphTruthMatch(const std::string name="PGraphTruthMatch");
    ~PGraphTruthMatch(){}
    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

  private:

    TTree* _tree;
    
    std::string _adc_img2d_prod;
    std::string _true_img2d_prod;
    std::string _reco_pgraph_prod;
    std::string _reco_pixel_prod;

    int _run;
    int _subrun;
    int _event;
    int _entry; 

    int _vtxid;
    int _vtx_on_nu;
    std::vector<int> _vtx_on_nu_v;

    std::vector<std::vector<int> > _par_id_vv;
    std::vector<int> _par_npx_v;

    float _vtx_x;
    float _vtx_y;
    float _vtx_z;


    void ClearVertex();


  };

  class PGraphTruthMatchProcessFactory : public ProcessFactoryBase {
  public:
    PGraphTruthMatchProcessFactory() { ProcessFactory::get().add_factory("PGraphTruthMatch",this); }
    ~PGraphTruthMatchProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new PGraphTruthMatch(instance_name); }
  };
  
}

#endif
