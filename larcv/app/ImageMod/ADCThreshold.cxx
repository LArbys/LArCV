#ifndef __ADCTHRESHOLD_CXX__
#define __ADCTHRESHOLD_CXX__

#include "ADCThreshold.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static ADCThresholdProcessFactory __global_ADCThresholdProcessFactory__;

  ADCThreshold::ADCThreshold(const std::string name)
    : ProcessBase(name)
  {}
    
  void ADCThreshold::configure(const PSet& cfg)
  {
    _image_producer = cfg.get< std::string        > ( "ImageProducer" );
    _min_adc_v      = cfg.get< std::vector<float> > ( "MinADC"        );
    _max_adc_v      = cfg.get< std::vector<float> > ( "MaxADC"        );
    if(_min_adc_v.size() != _max_adc_v.size()) {
      LARCV_CRITICAL() << "Length of configuration parameter array is not same!" << std::endl;
      throw larbys();
    }
  }

  void ADCThreshold::initialize()
  {}

  bool ADCThreshold::process(IOManager& mgr)
  {
    auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));

    std::vector<larcv::Image2D> image_v;
    event_image->Move(image_v);

    if(image_v.size() != _min_adc_v.size()) {
      LARCV_CRITICAL() << "# planes in the image and configured parameter array length do not match!" << std::endl;
      throw larbys();
    }
    
    for(size_t img_index=0; img_index<image_v.size(); ++img_index) {

      auto& img = image_v[img_index];

      auto min_adc = _min_adc_v[img_index];
      auto max_adc = _max_adc_v[img_index];
      
      auto const& adc_v = img.as_vector();
      _buffer.resize(adc_v.size(),0.);

      for(size_t i=0; i<adc_v.size(); ++i) {
	if(adc_v[i] < min_adc) _buffer[i] = 0.;
	else if(adc_v[i] > max_adc) _buffer[i] = max_adc;
	else _buffer[i] = adc_v[i];
      }

      img.copy(0,0,_buffer,_buffer.size());
    }

    event_image->Emplace(std::move(image_v));

    return true;
  }

  void ADCThreshold::finalize()
  {}

}
#endif
