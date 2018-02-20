#ifndef __SSNETEVENTANA_H__
#define __SSNETEVENTANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include "TTree.h"

namespace larcv {

  class SSNetEventAna : public ProcessBase {

  public:
    
    SSNetEventAna(const std::string name="SSNetEventAna");
    ~SSNetEventAna(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();
    
  private:

    void Reset();

    TTree* _tree;

    std::string _ev_img2d_prod;
    std::string _ev_score0_prod;
    std::string _ev_score1_prod;
    std::string _ev_score2_prod;
    std::string _ev_trk2d_prod;
    std::string _ev_shr2d_prod;

    float _threshold;

    int _run;
    int _subrun;
    int _event;
    int _entry;
    
    float _n_trk_pixel0;
    float _n_trk_pixel1;
    float _n_trk_pixel2;

    float _avg_trk_pixel_val0;
    float _avg_trk_pixel_val1;
    float _avg_trk_pixel_val2;

    float _std_trk_pixel_val0;
    float _std_trk_pixel_val1;
    float _std_trk_pixel_val2;

    float _tot_trk_pixel_val0;
    float _tot_trk_pixel_val1;
    float _tot_trk_pixel_val2;

    float _avg_trk_score0;
    float _avg_trk_score1;
    float _avg_trk_score2;

    float _std_trk_score0;
    float _std_trk_score1;
    float _std_trk_score2;

    float _n_shr_pixel0;
    float _n_shr_pixel1;
    float _n_shr_pixel2;

    float _avg_shr_pixel_val0;
    float _avg_shr_pixel_val1;
    float _avg_shr_pixel_val2;

    float _std_shr_pixel_val0;
    float _std_shr_pixel_val1;
    float _std_shr_pixel_val2;

    float _tot_shr_pixel_val0;
    float _tot_shr_pixel_val1;
    float _tot_shr_pixel_val2;

    float _avg_shr_score0;
    float _avg_shr_score1;
    float _avg_shr_score2;

    float _std_shr_score0;
    float _std_shr_score1;
    float _std_shr_score2;

    float _n_adc_pixel0;
    float _n_adc_pixel1;
    float _n_adc_pixel2;

    float _avg_adc_pixel_val0;
    float _avg_adc_pixel_val1;
    float _avg_adc_pixel_val2;

    float _std_adc_pixel_val0;
    float _std_adc_pixel_val1;
    float _std_adc_pixel_val2;

    float _tot_adc_pixel_val0;
    float _tot_adc_pixel_val1;
    float _tot_adc_pixel_val2;
  };

  class SSNetEventAnaProcessFactory : public ProcessFactoryBase {
  public:
    SSNetEventAnaProcessFactory() { ProcessFactory::get().add_factory("SSNetEventAna",this); }
    ~SSNetEventAnaProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new SSNetEventAna(instance_name); }
  };

}

#endif

