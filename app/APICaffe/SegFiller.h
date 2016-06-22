/**
 * \file SegFiller.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class SegFiller
 *
 * @author vic
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __SEGFILLER_H__
#define __SEGFILLER_H__

#include "Processor/ProcessFactory.h"
#include "SegDatumFillerBase.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class SegFiller ... these comments are used to generate
     doxygen documentation!
  */
  class SegFiller : public SegDatumFillerBase {

  public:
    
    /// Default constructor
    SegFiller(const std::string name="SegFiller");
    
    /// Default destructor
    ~SegFiller(){}

    void child_configure(const PSet&);

    void child_initialize();

    void child_batch_begin();

    void child_batch_end();

    void child_finalize();

    void set_dimension(const std::vector<larcv::Image2D>&);

    void fill_entry_data(const std::vector<larcv::Image2D>&,const std::vector<larcv::Image2D>&);

    const std::vector<bool>& mirrored() const { return _mirrored; }


  private:
    void assert_dimension(const std::vector<larcv::Image2D>&);

    std::vector<size_t> _slice_v;
    size_t _max_ch;
    std::vector<float> _max_adc_v;
    std::vector<float> _min_adc_v;
    std::vector<size_t> _caffe_idx_to_img_idx;
    std::vector<size_t> _mirror_caffe_idx_to_img_idx;
    std::vector<size_t> _roitype_to_class;
    std::vector<bool>   _mirrored;
    bool   _mirror_image;
    double _adc_gaus_mean;
    double _adc_gaus_sigma;
    bool _adc_gaus_pixelwise;
    bool _crop_image;
    bool _randomize_crop;
    int _crop_cols;
    int _crop_rows;
  };

  /**
     \class larcv::SegFillerFactory
     \brief A concrete factory class for larcv::SegFiller
  */
  class SegFillerProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    SegFillerProcessFactory() { ProcessFactory::get().add_factory("SegFiller",this); }
    /// dtor
    ~SegFillerProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new SegFiller(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

