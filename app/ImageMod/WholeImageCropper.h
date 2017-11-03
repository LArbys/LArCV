/**
 * \file WholeImageCropper.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class WholeImageCropper
 *
 * @author drinkingkazu
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __WHOLEIMAGECROPPER_H__
#define __WHOLEIMAGECROPPER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/Image2D.h"
namespace larcv {

  /**
     \class ProcessBase
     User defined class WholeImageCropper ... these comments are used to generate
     doxygen documentation!
  */
  class WholeImageCropper : public ProcessBase {

  public:
    
    /// Default constructor
    WholeImageCropper(const std::string name="WholeImageCropper");
    
    /// Default destructor
    ~WholeImageCropper(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    const larcv::Image2D& get_image() const { return _image; }

    const std::vector<larcv::Image2D>& get_cropped_image() const { return _cropped_v; }

    size_t target_rows() const { return _target_rows; }
    size_t target_cols() const { return _target_cols; }

  private:

    std::string _image_producer;
    std::string _roi_producer;
    size_t _target_rows;
    size_t _target_cols;
    size_t _target_ch;
    larcv::Image2D _image;
    std::vector<larcv::Image2D> _cropped_v;

  };

  /**
     \class larcv::WholeImageCropperFactory
     \brief A concrete factory class for larcv::WholeImageCropper
  */
  class WholeImageCropperProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    WholeImageCropperProcessFactory() { ProcessFactory::get().add_factory("WholeImageCropper",this); }
    /// dtor
    ~WholeImageCropperProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new WholeImageCropper(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

