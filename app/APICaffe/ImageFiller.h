/**
 * \file ImageFiller.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ImageFiller
 *
 * @author Rui A.
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __MULTICLASSFILLER_H__
#define __MULTICLASSFILLER_H__

#include "Processor/ProcessFactory.h"
#include "ImageFillerBase.h"
#include "RandomCropper.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class ImageFiller ... these comments are used to generate
     doxygen documentation!
  */
  class ImageFiller : public ImageFillerBase {

  public:
    
    /// Default constructor
    ImageFiller(const std::string name="ImageFiller");
    
    /// Default destructor
    ~ImageFiller(){}

    void child_configure(const PSet&);

    void child_initialize();

    void child_batch_begin();

    void child_batch_end();

    void child_finalize();

    const std::vector<bool>& mirrored() const { return _mirrored; }

    const std::vector<bool>& transposed() const { return _transposed; }

    const std::vector<int> dim(bool image=true) const;

  protected:

    void fill_entry_data(const EventBase* image_data);
    
    size_t compute_image_size(const EventBase* image_data);

  private:

    void assert_dimension(const std::vector<larcv::Image2D>&);

    size_t _rows;
    size_t _cols;
    size_t _num_channels;
    std::vector<size_t> _slice_v;
    size_t _max_ch;
    std::vector<size_t> _caffe_idx_to_img_idx;
    std::vector<size_t> _mirror_caffe_idx_to_img_idx;
    std::vector<size_t> _transpose_caffe_idx_to_img_idx;
    std::vector<size_t> _roitype_to_class;
    std::vector<bool>   _mirrored;
    std::vector<bool>   _transposed;
    bool _mirror_image;
    bool _crop_image;
    bool _transpose_image;
    bool _use_norm;
    RandomCropper _cropper;

  };

  /**
     \class larcv::ImageFillerFactory
     \brief A concrete factory class for larcv::ImageFiller
  */
  class ImageFillerProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    ImageFillerProcessFactory() { ProcessFactory::get().add_factory("ImageFiller",this); }
    /// dtor
    ~ImageFillerProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new ImageFiller(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

