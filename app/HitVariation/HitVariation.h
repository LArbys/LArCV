#ifndef __HITVARIATION__
#define __HITVARIATION__

#include <string>
#include <vector>

// larcv
#include "DataFormat/IOManager.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ImageMeta.h"
#include "DataFormat/Pixel2D.h"

// ROOT
#include "TRandom3.h"

namespace larcv {

 	class HitVariation {

 		HitVariation() {}; // private default constructor. Needed For ROOT bindings, but User cannot make class this way.

 	public:

   	HitVariation( std::string configuration_filename );
    virtual ~HitVariation();

    // Parameters are collected here in order to keep track of them
    // mostly examples. Matt feel free to change this.
    // Parameters set in constructor
    struct HVParameters {
    	double overall_std;
    	double pixel_std;
    	double translation_std;
    	double amplitude_std;
    	double width_std;
    	std::string pixel_tree_name;
    	std::string originalimg_tree_name;
    	std::string roi_label_tree_name;
    	int numplanes;
        bool make_random_image;
    } m_config;

   IOManager& GetIOManager() { return *m_ioman; }

   // primary method
   void GenerateImages( const int num_images, const std::vector<double>& parameters, 
			std::vector< std::vector<larcv::Image2D> >& output_orig,
			std::vector< std::vector<larcv::Image2D> >& output_manip,
			std::vector<int>& labels, std::vector<int>& entrynumbers );

   // supporting methods:
   // Below is for Matt's code
   larcv::Image2D MakeImageFromHits( const std::vector<larcv::Pixel2D>& pixels, const larcv::ImageMeta& meta, const std::vector<double>& parameters );
   // Below is just an example
   std::vector<larcv::Image2D> MakeRandomImage( const std::vector<larcv::Pixel2D>& pixels, const larcv::ImageMeta& meta, const std::vector<double>& parameters );

  protected:
  	// IOManager: this is the interface to the LArCV files
    IOManager* m_ioman;
    TRandom3* m_rng;
    long m_seed;
    int m_nentries;

  };

}

#endif
