/**
 * \file MultiClassFiller.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class MultiClassFiller
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __MULTICLASSFILLER_H__
#define __MULTICLASSFILLER_H__

#include "Processor/ProcessFactory.h"
#include "DatumFillerBase.h"
#include "RandomCropper.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class MultiClassFiller ... these comments are used to generate
     doxygen documentation!
  */
  class MultiClassFiller : public DatumFillerBase {

  public:
    
    /// Default constructor
    MultiClassFiller(const std::string name="MultiClassFiller");
    
    /// Default destructor
    ~MultiClassFiller(){}

    void child_configure(const PSet&);

    void child_initialize();

    void child_batch_begin();

    void child_batch_end();

    void child_finalize();

    const std::vector<bool>& mirrored() const { return _mirrored; }

    const std::vector<int> dim(bool image=true) const;

  protected:

    void fill_entry_data(const EventBase* image_data, 
			 const EventBase* label_data,
			 const EventBase* weight_data);

    size_t compute_image_size(const EventBase* image_data);

    size_t compute_label_size(const EventBase* label_data);

  private:

    void assert_dimension(const std::vector<larcv::Image2D>&);

    size_t _rows;
    size_t _cols;
    size_t _num_channels;
    std::vector<size_t> _slice_v;
    size_t _max_ch;
    std::vector<size_t> _caffe_idx_to_img_idx;
    std::vector<size_t> _mirror_caffe_idx_to_img_idx;
    std::vector<size_t> _roitype_to_class;
    std::vector<bool>   _mirrored;
    bool _mirror_image;
    bool _crop_image;

    RandomCropper _cropper;

  };

  /**
     \class larcv::MultiClassFillerFactory
     \brief A concrete factory class for larcv::MultiClassFiller
  */
  class MultiClassFillerProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    MultiClassFillerProcessFactory() { ProcessFactory::get().add_factory("MultiClassFiller",this); }
    /// dtor
    ~MultiClassFillerProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new MultiClassFiller(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

