#ifndef __UBSPLITDETECTOR_CXX__
#define __UBSPLITDETECTOR_CXX__

#include <ctime>

#include "UBSplitDetector.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventImage2D.h"

#ifdef HAS_LARLITE
//larlite
#include "LArUtil/Geometry.h"
#endif

// ROOT TRandom3
#include "TRandom3.h"

namespace larcv {

  static UBSplitDetectorProcessFactory __global_UBSplitDetectorProcessFactory__;

  float UBSplitDetector::elapsed_genbbox = 0;
  float UBSplitDetector::elapsed_crop = 0;
  float UBSplitDetector::elapsed_alloc = 0;
  float UBSplitDetector::elapsed_fraccheck = 0;
  float UBSplitDetector::elapsed_save = 0;
  int   UBSplitDetector::num_calls = 0;
  
  UBSplitDetector::UBSplitDetector(const std::string name)
    : ProcessBase(name)
  {
    _num_expected_crops = -1;
    _numcrops_changed = false;

    elapsed_genbbox = 0;
    elapsed_crop = 0;
    elapsed_alloc = 0;
    elapsed_fraccheck = 0;
    elapsed_save = 0;
    num_calls = 0;
#ifndef HAS_LARLITE
    throw std::runtime_error("UBSplitDetector must be compiled with larlite. Must rebuild.");
#endif
  }

  void UBSplitDetector::configure(const PSet& cfg)
  {
    // operating parameters
    // name of tree from which to get ADC images and their meta
    _input_producer        = cfg.get<std::string>("InputProducer");

    // name of producer to store output bounding boxes for each crop
    // all bboxes stored in unrolled vector in (u,v,y) plane order
    // i.e. (bbox0_u, bbox0_v, bbox0_y, bbox1_u, bbox1_v, bbox1_y, ...)
    _output_bbox_producer  = cfg.get<std::string>("OutputBBox2DProducer");

    // we can ask this module to do the crop for us. intended to work
    // with ADC images. other truth-label images, you are on your own
    _enable_img_crop       = cfg.get<bool>("CropInModule",true);

    // if we do the crop, this is used to name the output tree
    _output_img_producer   = cfg.get<std::string>("OutputCroppedProducer");

    // set dimensions of bounding box
    // pixel height will set the tick bounds of the image
    _box_pixel_height      = cfg.get<int>("BBoxPixelHeight",512);

    // this parameter sets the width of the image
    // it is the width in the Z dimension that defines visible Y-wires
    // the U and V wires saved in the image are chosen to completely cover
    // this range of Y-wires. This means, the range of filled pixels in U,V
    // will be larger than Y. Y-wires are centered and the unused pixel columns
    // are blanked out
    _covered_z_width       = cfg.get<int>("CoveredZWidth",310);

    // enforces a maximum picture width. we clip the edges of the
    // Y-overlapping U,V wires
    _box_pixel_width       = cfg.get<int>("BBoxPixelWidth",832);

    // by default, we leave ends of y-image blank, but we have option to fill completely
    _complete_y_crop      = cfg.get<bool>("FillCroppedYImageCompletely",false);

    // dump a png for debuggin
    _debug_img             = cfg.get<bool>("DebugImage",false);

    // we will split the detector completely (if not in random mode)
    // this caps the number of images. useful for debug, as this is
    // a fairly slow module
    _max_images            = cfg.get<int>("MaxImages",-1);

    // we can also choose to randomly crop within the detector
    // random (t,z) coordinate will be chosen and that location
    // will be cropped
    _randomize_crops       = cfg.get<bool>("RandomizeCrops",false);

    // max number of bounding boxes to generate
    // but we only save the number of images according to MaxImages
    _randomize_attempts    = cfg.get<int>("MaxRandomAttempts",10);

    // max fraction of pixels with above threshold values
    // else we don't keep it
    // if value is 0 or less, then we do not enforce this cut
    _randomize_minfracpix  = cfg.get<float>("MinFracPixelsInCrop",-1.0);
    if (!_randomize_crops) {
      _randomize_minfracpix = -1.0; // only use filter for random cropping
    }

    // verbosity
    set_verbosity( (larcv::msg::Level_t)cfg.get<int>("Verbosity",2) );
  }

  void UBSplitDetector::initialize()
  {}

  bool UBSplitDetector::process(IOManager& mgr)
  {
    // we split the full detector image into 3D subpieces

    // ---------------------------------------------------------------
    // get data
    auto input_image  = (larcv::EventImage2D*)(mgr.get_data(larcv::kProductImage2D, _input_producer));
    if (!input_image) {
      LARCV_CRITICAL() << "No Image2D found with a name: " << _input_producer << std::endl;
      throw larbys();
    }
    const std::vector< larcv::Image2D >& img_v = input_image->Image2DArray();

    larcv::EventROI*  output_bbox     = (larcv::EventROI*)     mgr.get_data(larcv::kProductROI,     _output_bbox_producer);
    larcv::EventImage2D* output_imgs  = (larcv::EventImage2D*) mgr.get_data(larcv::kProductImage2D, _output_img_producer);
    // we reset the output variables
    // clear the list of bounding boxes
    output_bbox->clear();

    // we don't want to reallocate a large number of images each time we run this process
    //  if we can avoid it. random crops we have to, but for non-random crops, the subimages should be
    //  the same.    
    if ( _randomize_crops ) {
      // we clear for randomized crops and live with realloc...
      output_imgs->clear();
    }
    else {
      LARCV_DEBUG() << "reset current images. num=" << output_imgs->Image2DArray().size() << std::endl;
      mgr.donot_clear_product(larcv::kProductImage2D,_output_img_producer); // we take over clearing this container
      std::vector<larcv::Image2D> temp_v;
      output_imgs->Move( temp_v );
      for ( auto& img : temp_v )
	img.paint(0.0);
      output_imgs->Emplace( std::move(temp_v) );
    }
    // ----------------------------------------------------------------
    

    std::vector< larcv::ROI > outbbox_v;
    std::vector< larcv::Image2D > outimg_v;
    output_imgs->Move( outimg_v );

    bool status = process( output_imgs->Image2DArray(), outimg_v, outbbox_v );

    output_imgs->Emplace( std::move(outimg_v) );
    output_bbox->Emplace( std::move(outbbox_v) );
    
    return status;
  }

  bool UBSplitDetector::process( const std::vector<larcv::Image2D>& img_v, std::vector<larcv::Image2D>& outimg_v, std::vector<larcv::ROI>& outbbox_v)
  {
    // we split the full detector image into 3D subpieces

    // ---------------------------------------------------------------
    // get data
  

    // first define the lattice of 3D points
    // set lattice (y,z) pitch using width of image

    // --- image parameters ---
    // we aim to make an image where all y-charge has a partner to match against
    //const float dudz = 0.3*2;
    //const float dudy = 0.3*2/sqrt(3);
    //const float detheight = 117.0*2.0;
    int zwidth     = _covered_z_width;

    // --- x/tick divisions ----

    const larcv::ImageMeta& meta = img_v.front().meta();
    float dtick = _box_pixel_height*meta.pixel_height();

    float dtickimg = meta.max_y()-meta.min_y() - dtick;
    int nt = dtickimg/(0.5*dtick);
    if ( fabs(nt*0.5*dtick-dtickimg)>0.5 ) nt++;
    float tstep  = -dtickimg/nt;
    float startt = meta.max_y() - dtick/2;

    // --- z divisions ---------

    int zcols      = img_v.front().meta().cols();

    int zend       = zcols-_box_pixel_width*meta.pixel_width()/2;
    int zstart     = _box_pixel_width*meta.pixel_width()/2;
    int zspan      = zend-zstart;
    int nz         = zspan/(zwidth/2);
    if ( abs( (zwidth/2)*nz - zspan )!=0 )
      nz++;
    float zstep = float(zspan)/float(nz);

    // store cropped coordinates in m_lattice;
    // to help performance, we avoid reallocating/refilling vectors
    // which won't change in size
    //std::vector< std::vector<int> > lattice;

    if ( !_randomize_crops ) {
      // crop in lattice pattern through detector
      LARCV_DEBUG() << "Lattice Crop" << std::endl;
      LARCV_DEBUG() << "dtickimg=" << dtickimg << std::endl;
      LARCV_DEBUG() << "dtick=" << dtick << std::endl;
      LARCV_DEBUG() << "tstep=" << tstep << std::endl;
      LARCV_DEBUG() << "nt,nz: " << nt  << " " << nz << std::endl;
      LARCV_DEBUG() << "start (z,t): (" << zstart << ", " << startt << ")" << std::endl;

      int ncrops = (nz+1)*(nt+1);
      // did the number of crops changed?
      if ( ncrops!=_num_expected_crops ) {
	_num_expected_crops = ncrops;
	_numcrops_changed = true;
	m_lattice.clear();
	m_lattice.reserve( ncrops );
	outimg_v.clear(); // must change, so we clear
	// std::vector<larcv::Image2D> reserve_v(ncrops*3);
	// output_imgs->Emplace( std::move(reserve_v) );;
	LARCV_INFO() << "setting number of expected crops. ncrops=" << ncrops << " outsize=" << outimg_v.size() << std::endl;
      }
      else {
	_numcrops_changed = false;
	LARCV_DEBUG() << "no change in number of expected crops. "
		      << " ncrops=" << ncrops
		      << " num_expected_crops=" << _num_expected_crops
		      << " numcrops_changed_flag=" << _numcrops_changed
		      << " outsize=" << outimg_v.size() << std::endl;
      }
      
      if ( _numcrops_changed ) {
	// if the number of crops changed, we need to refill the m_lattice vector
	for (int it=0; it<=nt; it++) {
	
	  float tmid = startt + it*tstep;
	  
	  for (int iz=0; iz<=nz; iz++) {
	    
	    float zwire = zstart + zstep*iz;
	    std::clock_t begin = std::clock();
	    std::vector<int> crop_coords = defineImageBoundsFromPosZT( zwire, tmid, zwidth, dtick,
								       _box_pixel_width, _box_pixel_height,
								       img_v );
	    m_lattice.emplace_back( std::move(crop_coords) );
	    std::clock_t end = std::clock();
	    elapsed_genbbox += double(end - begin) / CLOCKS_PER_SEC;
	    
	  }
	}
      }
      LARCV_INFO() << "Full Image split into " << m_lattice.size() << " subimages" << std::endl;
      
    }
    else {
      // random cropping
      // number of crops determined by fcl file, so never changes
      _numcrops_changed   = false;
      _num_expected_crops = _randomize_attempts;
      
      // but we must refill the cropping coordinates each time
      m_lattice.clear();
      m_lattice.reserve( _randomize_attempts );
      
      TRandom3 rand(time(NULL));
      for (int iatt=0; iatt<_randomize_attempts; iatt++) {

	float z;
	if ( !_complete_y_crop )
	  z = zstart + zspan*rand.Uniform();
	else
	  z = _box_pixel_width + (zcols-2*_box_pixel_width)*rand.Uniform();

	float t = startt + dtickimg*rand.Uniform();

	std::vector<int> crop_coords = defineImageBoundsFromPosZT( z, t, zwidth, dtick,
								   _box_pixel_width, _box_pixel_height,
								   img_v );
	m_lattice.emplace_back( std::move(crop_coords) );
      }
      LARCV_INFO() << "Num of randomized cropping points generated: " << m_lattice.size() << std::endl;
    }

    // debug
    // we track coverage of crops by marking up an image
    //std::vector<larcv::Image2D> coverage_v;
    if ( _debug_img ) {
      if ( m_coverage_v.size()!=img_v.size() ) {
	// coverage vector does not match the input image vector size, update it
	m_coverage_v.clear();
	for ( int p=0; p<3; p++) {
	  larcv::Image2D cov( img_v[p].meta() );
	  cov.paint(0.0);
	  m_coverage_v.emplace_back( std::move(cov) );
	}
      }
      else {
	// zero it out (but don't reallocate)
	for ( int p=0; p<3; p++)
	  m_coverage_v[p].paint(0.0);
      }
    }

    // ---------------------------------------------
    // LATTICE LOOP
    // ---------------------------------------------
    // create bounding boxes around lattice points
    int nfilled = 0;
    int nrejected = 0;
    bool copy_imgs = true;
    if ( _enable_img_crop && _randomize_crops ) {
      // for randomization, since we do not know how many images to put into container,
      // we have to realloc (as opposed to copying data over)
      copy_imgs = false;
    }
    LARCV_DEBUG() << "Create ROI from lattice points" << std::endl;
    for ( auto const& cropcoords : m_lattice ) {

      if ( _max_images>0 && _max_images<=nfilled )
	break;

      int y1 = cropcoords[0];
      int y2 = cropcoords[1];
      int u1 = cropcoords[2];
      int u2 = cropcoords[3];
      int v1 = cropcoords[4];
      int v2 = cropcoords[5];
      int t1 = cropcoords[6];
      int t2 = cropcoords[7];

      std::clock_t begin = std::clock();
      LARCV_DEBUG() << "define one lattice point ROI" << std::endl;
      larcv::ROI bbox_vec = defineBoundingBoxFromCropCoords( img_v, _box_pixel_width, _box_pixel_height,
							     t1, t2, u1, u2, v1, v2, y1, y2 );
      std::clock_t end = std::clock();      
      elapsed_genbbox += double(end - begin) / CLOCKS_PER_SEC;

      // we have the option to fill images from these bounding boxes
      // issues:
      //   for simple splitting of image, we can avoid reallocation by overwriting pixel values
      //   for random cropping or filtering, the size changes. we resize to avoid as many reallocations as possible
      bool filledimg = false;
      if ( _enable_img_crop ) {
	if ( copy_imgs && ( _numcrops_changed || outimg_v.size()!=_num_expected_crops*img_v.size() ) ) {
	  // we need to create image to copy to
	  // the intention is for this to only run once
	  std::clock_t begin = std::clock();
	  //std::vector<larcv::Image2D> tmp;
	  //output_imgs->Move(tmp);
	  LARCV_DEBUG() << "output image container has " << outimg_v.size() 
			<< " images while we need " << _num_expected_crops*img_v.size() << std::endl;
	  for ( size_t ip=0; ip<img_v.size(); ip++ ) {

	    // larcv2 meta
	    // larcv::ImageMeta planecrop( bbox_vec[ip].min_x(), bbox_vec[ip].min_y(), bbox_vec[ip].max_x(), bbox_vec[ip].max_y(),
	    // 				(int)(bbox_vec[ip].height()/img_v[ip].meta().pixel_height()),
	    // 				(int)(bbox_vec[ip].width()/img_v[ip].meta().pixel_width()),
	    // 				img_v[ip].meta().id() );

	    // larcv1 meta
	    const larcv::ImageMeta& planecrop = bbox_vec.BB(ip);

	    
	    larcv::Image2D imgcrop( planecrop );
	    imgcrop.paint(0.0);
	    outimg_v.emplace_back( std::move(imgcrop) );
	  }
	  //output_imgs->Emplace( std::move(tmp) );
	  std::clock_t end = std::clock();
	  elapsed_alloc += double(end - begin) / CLOCKS_PER_SEC;
	  LARCV_DEBUG() << "Created image for copying values. Total number=" << outimg_v.size() << std::endl;
	}
	LARCV_DEBUG() << "crop using bbox2d, reuse image. nfilled=" << nfilled
		      << " copy_imgs=" << copy_imgs
		      << " numcrops_changed=" << _numcrops_changed
		      << " output_imgs_size=" << outimg_v.size()
		      << std::endl;
	begin = std::clock();
	filledimg = cropUsingBBox2D( bbox_vec, img_v, y1, y2, _complete_y_crop, _randomize_minfracpix, nfilled*3, copy_imgs, outimg_v );
	end = std::clock();
	elapsed_crop += double(end - begin) / CLOCKS_PER_SEC;
      }

      if ( filledimg || !_enable_img_crop ) {
	nfilled ++;
	//output_bbox->Emplace( std::move(bbox_vec) );
	outbbox_v.emplace_back( std::move(bbox_vec) );
      }
      else {
	nrejected ++;
      }

    }///end of loop over lattice

    LARCV_DEBUG() << "Number of cropped images: " << outimg_v.size() << std::endl;
    LARCV_DEBUG() << "Number of cropped images per plane: " << outimg_v.size()/3 << std::endl;
    LARCV_INFO()  << "BBoxes gen'ed=" << m_lattice.size() << " filled=" << nfilled << " rejected=" << nrejected << std::endl;

    // if ( _debug_img ) {
    //   auto outev_coverage = (larcv::EventImage2D*)(mgr.get_data("image2d", "coverage"));
    //   int nuncovered[3] = {0};
    //   float meancoverage[3] = {0};
    //   for (int p=0; p<3; p++) {
    // 	int maxc = 3456;
    // 	int maxr = img_v[p].meta().rows();
    // 	if ( p<2 )
    // 	  maxc = 2400;
    // 	for (int r=0; r<(int)img_v[p].meta().rows(); r++) {
    // 	  for (int c=0; c<maxc; c++) {
    // 	    if ( coverage_v[p].pixel(r,c)<0.5 )
    // 	      nuncovered[p]++;
    // 	    meancoverage[p] += coverage_v[p].pixel(r,c)/float(maxc*maxr);
    // 	  }
    // 	}
    // 	LARCV_INFO() << "plane " << p << ": uncovered=" << nuncovered[p] << "  meancoverage=" << meancoverage[p] << std::endl;
    // 	outev_coverage->Emplace( std::move(coverage_v[p]) );
    //   }
    // }
    num_calls++;
    printElapsedTime();
    
    return true;
  }

  larcv::ROI UBSplitDetector::defineBoundingBoxFromCropCoords( const std::vector<larcv::Image2D>& img_v,
							       const int box_pixel_width, const int box_pixel_height,
							       const int t1, const int t2,
							       const int u1, const int u2,
							       const int v1, const int v2,
							       const int y1, const int y2) {

    // takes pre-defined image bounds on all 3 planes (given in min/max row/col)
    // note, box_pixel_width and box_pixel_height are meant to be the same
    // variable as member variables _box_pixel_width and _box_pixel_height.
    // we pass them here in order to make this function static, so it can be used as a stand-alone function.

    // input
    // ------
    // img_v: source ADC images from which we are cropping
    // box_pixel_width: corresponds to _box_pixel_width
    // box_pixel_height: corresponds to _box_pixel_height
    // (x)1, (x)2, where x=t,u,v,y
    // row bounds (t) and col bounds (u,v,y) max-exclusive [x1,x2)
    //
    // output
    // -------
    // (return) vector of bounding boxes defined for (u,v,y) plane

    larcv::ROI bbox_vec; // we create one for each plane
    //bbox_vec.reserve(img_v.size());

    const larcv::ImageMeta& meta = img_v.front().meta(); // it is assumed that time-coordinate meta same between planes

    // define tick and row bounds
    int nrows = box_pixel_height;
    float mint = meta.pos_y(t2);
    float maxt = meta.pos_y(t1);

    // we crop an image with W x H = maxdu x _box_pixel_height
    // we embed in the center, the Y-plane source image with zwidth across
    // we crop the entire range for the U or V plane, the target images

    //LARCV_DEBUG() << "Defining bounding boxes" << std::endl;

    // prepare the u-plane
    const larcv::ImageMeta& umeta = img_v[0].meta();
    float minu = umeta.pos_x( u1 );
    float maxu = umeta.pos_x( u2 );
    //larcv::BBox2D bbox_u( minu, mint, maxu, maxt, img_v[0].meta().id() );
    //larcv::ImageMeta metacropu( minu, mint, maxu, maxt, nrows, box_pixel_width, img_v[0].meta().id() );
    larcv::ImageMeta metacropu( maxu-minu, maxt-mint, (maxt-mint)/umeta.pixel_height(), (maxu-minu)/umeta.pixel_width(), minu, maxt, 0 );

    // prepare the v-plane
    const larcv::ImageMeta& vmeta = img_v[1].meta();
    float minv = vmeta.pos_x( v1 );
    float maxv = vmeta.pos_x( v2 );
    //larcv::BBox2D bbox_v( minv, mint, maxv, maxt, img_v[1].meta().id() );
    larcv::ImageMeta metacropv( maxv-minv, maxt-mint, (maxt-mint)/vmeta.pixel_height(), (maxv-minv)/vmeta.pixel_width(), minv, maxt, 1 );
    
    // prepare the y-plane
    // we take the narrow range and try to put it in the center of the full y-plane image
    const larcv::ImageMeta& ymeta = img_v[2].meta();
    int ycenter = (y1+y2)/2;
    int ycmin   = ycenter - (int)metacropu.cols()/2;
    int ycmax   = ycmin + (int)metacropu.cols();
    float miny = 0;
    float maxy = 0;
    if ( ycmin>=0 && ycmax<(int)ymeta.cols() ) {
      miny = ymeta.pos_x( ycmin );
      maxy = ymeta.pos_x( ycmax );
    }
    else if ( ycmin<ymeta.min_x() ) {
      miny = ymeta.min_x();
      maxy = ymeta.pos_x( 0+metacropu.cols() );
    }
    else if ( ycmax>=(int)ymeta.cols() ) {
      maxy = ymeta.max_x()-1;
      miny = ymeta.pos_x( ymeta.cols()-metacropu.cols()-1 );
    }
    // larcv::ImageMeta crop_yp( miny, mint, maxy, maxt,
    // 			      (maxt-mint)/ymeta.pixel_height(),
    // 			      ycmax-ycmin,
    // 			      ymeta.id() );
    larcv::ImageMeta crop_yp( maxy-miny, maxt-mint, 
    			      (maxt-mint)/ymeta.pixel_height(),
    			      (maxy-miny)/ymeta.pixel_width(),			      
			      miny, maxt, ymeta.plane() );
    //larcv::BBox2D bbox_y( miny, mint, maxy, maxt, ymeta.id() );

    // bbox_vec.emplace_back( std::move(bbox_u) );
    // bbox_vec.emplace_back( std::move(bbox_v) );
    // bbox_vec.emplace_back( std::move(bbox_y) );
    std::vector<larcv::ImageMeta> bb;
    bb.emplace_back( std::move(metacropu) );
    bb.emplace_back( std::move(metacropv) );
    bb.emplace_back( std::move(crop_yp) );    
    bbox_vec.SetBB( bb );
      
    return bbox_vec;

  }

  bool UBSplitDetector::cropUsingBBox2D( const larcv::ROI& bbox_vec,
					 const std::vector<larcv::Image2D>& img_v,
					 const int y1, const int y2, bool fill_y_image,
					 const float minpixfrac,
					 const int first_outidx,
					 const bool copy_imgs,
					 larcv::EventImage2D& output_imgs ) {

    // get the vector of images
    std::vector<larcv::Image2D> outimg_v;
    output_imgs.Move( outimg_v );

    cropUsingBBox2D( bbox_vec, img_v, y1, y2, fill_y_image, minpixfrac, first_outidx, copy_imgs, outimg_v );
    
    // give pack the image vector
    output_imgs.Emplace( std::move(outimg_v) );
  }
  
  bool UBSplitDetector::cropUsingBBox2D( const larcv::ROI& bbox_vec,
					 const std::vector<larcv::Image2D>& img_v,
					 const int y1, const int y2, bool fill_y_image,
					 const float minpixfrac,
					 const int first_outidx,
					 const bool copy_imgs,
					 std::vector<larcv::Image2D>& outimg_v ) {

    // inputs
    // ------
    // bbox_v, vector of bounding boxes for (u,v,y)
    // img_v, source adc images
    // y1, y2: range of y-wires fully covered by U,V
    // fill_y_image: if true, we fill the entire cropped y-image.
    //               else we only fill the region of y-wires that
    //               are entirely covered by the u,v cropped images
    // minpixfrac: if value is >0, we enforce a minimum value on the
    //               number of pixels occupied in the Y-image
    // first_outidx: if copying images instead of adding images into a vector
    // copy_imgs: copy, don't create new image and fill
    //
    // outputs
    // --------
    // output_imgs, cropped output image2d instances filled into eventimage2d container
    //
    // note: we try to avoid repeated allocation/deallocation of images as this does not scale well
    // 

    // metas for the source images
    std::vector< larcv::ImageMeta > src_metas;
    for ( auto const& srcimg : img_v )
      src_metas.push_back( srcimg.meta() );
    
    // metas for the cropped images
    std::vector< larcv::ImageMeta > crop_metas;
    for ( size_t ip=0; ip<img_v.size(); ip++ ) {
      // larcv::ImageMeta planecrop( bbox_vec[ip].min_x(), bbox_vec[ip].min_y(), bbox_vec[ip].max_x(), bbox_vec[ip].max_y(),
      // 				  (int)(bbox_vec[ip].height()/src_metas[ip].pixel_height()),
      // 				  (int)(bbox_vec[ip].width()/src_metas[ip].pixel_width()),
      // 				  src_metas[ip].id() );
      const larcv::ImageMeta& planecrop = bbox_vec.BB(ip);
      crop_metas.push_back( planecrop );
    }

    // get the vector of images
    //std::vector<larcv::Image2D> outimg_v;
    //output_imgs.Move( outimg_v );
    //std::cout << "cropper: outimg_v size=" << outimg_v.size() << std::endl;

    // y-image crop
    larcv::Image2D  y_img_out_new;        // if we make a new img (we'll move to output container)
    larcv::Image2D* y_img_out_old = NULL; // if we xfer values (we refer to image already in container)
    
    if ( copy_imgs ) {
      // copy values, don't create new (ugh, so ugly)
      y_img_out_old = &(outimg_v.at( first_outidx+2 )); // y output image
      y_img_out_old->modifyMeta( crop_metas[2] ); // set the output meta
    }
    
    if ( fill_y_image ) {
      // we fill all y-columns will all values
      if ( copy_imgs ) {
	//std::cout << "copy region: meta=" << y_img_out_old->meta().dump() << " intended meta=" << crop_metas[2].dump() << std::endl;
	y_img_out_old->copy_region( img_v[2] ); // copy pixels within this region
      }
      else {
	std::clock_t begin = std::clock();
	larcv::Image2D newyimg( crop_metas[2] );
	std::clock_t end  = std::clock();
	elapsed_alloc += double(end-begin)/CLOCKS_PER_SEC;

	//begin = std::clock();		
	newyimg.copy_region( img_v[2] );
	std::swap( newyimg, y_img_out_new );
	//end = std::clock();
	//elapsed_crop += double(end-begin)/CLOCKS_PER_SEC;
	//y_img_out_new = img_v[2].crop( bbox_vec[2] ); // shitty?
      }
    }
    else {
      // we only fill y-wires that are fully covered by U,V wires

      // set first row
      int t1 = src_metas[2].row( bbox_vec.BB(2).max_y() );

      // get address of y-image we are going to fill
      // whether we use a new image, or the image in the existing output vector
      // is determined by the copy_imgs flag
      larcv::Image2D* y_target = NULL;
      if ( copy_imgs ) {
	y_img_out_old->paint(0.0);
	y_target = y_img_out_old;
      }
      else {
	larcv::Image2D ynew( crop_metas[2] );
	ynew.paint(0.0);
	std::swap( y_img_out_new, ynew );
	y_target = &y_img_out_new;
      }

      // finally, fill
      for (int c=0; c<(int)crop_metas[2].cols(); c++) {
	float cropx = crop_metas[2].pos_x(c);

	if ( cropx<y1 || cropx>=y2 )
	  continue;
	int cropc = src_metas[2].col(cropx);
	for (int r=0; r<(int)crop_metas[2].rows(); r++) {
	  //std::cout << "fill ytarget (" << r << "," << c << ")"  << std::endl;
	  y_target->set_pixel( r, c, img_v[2].pixel( t1+r, cropc ) );
	}
      }
    }

    float frac_occupied = 0.;
    bool saveimg = true;

    // if we enforce a minimum pixel fraction of the Y-plane
    // we evaulate that here, setting saveimg flag
    if ( minpixfrac>0 ) {
      larcv::Image2D* ycount = NULL;
      if ( copy_imgs )
	ycount = y_img_out_old;
      else
	ycount = &y_img_out_new;
      
      for (int row=0; row<(int)ycount->meta().rows(); row++) {
	for (int col=0; col<(int)ycount->meta().cols(); col++) {
	  if ( ycount->pixel(row,col)>10.0 )
	    frac_occupied+=1.0;
	}
      }
      frac_occupied /= float(ycount->meta().rows()*ycount->meta().cols());

      if ( frac_occupied<minpixfrac )
	saveimg = false;
    }


    if ( !saveimg )
      return false;
    

    // crop and store other planes
    if ( copy_imgs ) {
      // we xfer values to existing images in output_imgs
      larcv::Image2D& crop_up = outimg_v.at( first_outidx+0 );
      crop_up.modifyMeta( crop_metas[0] );
      crop_up.copy_region( img_v[0] );
    
      larcv::Image2D& crop_vp = outimg_v.at( first_outidx+1 );
      crop_vp.modifyMeta( crop_metas[1] );
      crop_vp.copy_region( img_v[1] );

      // we already xfer'd values to y-above
    }
    else {
      // we give new image to vector      
      //larcv::Image2D crop_up = img_v[0].crop( bbox_vec[0] );
      //larcv::Image2D crop_vp = img_v[1].crop( bbox_vec[1] );

      std::clock_t begin = std::clock();
      larcv::Image2D crop_up( crop_metas[0] );
      larcv::Image2D crop_vp( crop_metas[1] );
      std::clock_t end  = std::clock();
      elapsed_alloc += double(end-begin)/CLOCKS_PER_SEC;

      crop_up.copy_region( img_v[0] );
      crop_vp.copy_region( img_v[1] );      

      outimg_v.emplace_back( std::move(crop_up) );
      outimg_v.emplace_back( std::move(crop_vp) );
      outimg_v.emplace_back( std::move(y_img_out_new) );
      // output_imgs.Emplace( std::move(crop_up) );
      // output_imgs.Emplace( std::move(crop_vp) );      
      // output_imgs.Emplace( std::move(y_img_out_new) );
    }
    // give pack the image vector
    //output_imgs.Emplace( std::move(outimg_v) );
    return true;
  }

  std::vector<int> UBSplitDetector::defineImageBoundsFromPosZT( const float zwire, const float tmid, const float zwidth, const float dtick,
								const int box_pixel_width, const int box_pixel_height,
								const std::vector<larcv::Image2D>& img_v ) {

    // zwidth will be smaller than image size. that is because we are specificying range where we have complete overlap with U,V
    // however, we will expand around this region, filling edges of Y image with information

    const larcv::ImageMeta& meta = img_v.front().meta();
#ifdef HAS_LARLITE
    const larutil::Geometry* geo = larutil::Geometry::GetME();
#endif

    float t1 = tmid-0.5*dtick;
    float t2 = tmid+0.5*dtick;
    int r1,r2;
    try {
      r1 = meta.row( t2-0.5*meta.pixel_height() );
      if ( t1>meta.min_y() )	
	r2 = meta.row( t1 );
      else
	r2 = meta.row( t1+meta.pixel_height() )+1;
    }
    catch ( const std::exception& e ) {
      std::cout << __PRETTY_FUNCTION__ << "::" __FILE__ << ":" << __LINE__
		<< ": tick bounds outside image. tmid=" << tmid << " t1=" << t1 << " t2=" << t2 << ". min_t=" << meta.min_y() << " max_t=" << meta.max_y() << std::endl;
      throw e;
    }
    //std::cout << "tmid=" << tmid << " [" << t1 << "," << t2 << "] = [" << r1 << "," << r2 << "]" << std::endl;
    
    // fix tick bounds
    if ( r2-r1!=box_pixel_height ) {
      r1 = r2-box_pixel_height;
    }

    if ( r1<0 ) {
      r1 = 0;
      r2 = r1 + box_pixel_height;
    }
    if ( r2>(int)meta.rows() ) {
      r2 = (int)meta.rows();
      r1 = r2-meta.rows();
    }

    // set z range
    int zcol0 = zwire - zwidth/2;
    int zcol1 = zwire + zwidth/2;

    if ( zcol1>3455 ) {
      std::stringstream ss;
      ss << __PRETTY_FUNCTION__ << ":" << __FILE__ << "." << __LINE__ << ": zcol1 extends beyond the image boundary?" << std::endl;
      throw std::runtime_error( ss.str() );
      zcol0 -= (zcol1-3455);
      zcol1 = 3455;
    }

    if ( zcol0 < meta.min_x() || zcol1 >= meta.max_x() ) {
      std::stringstream ss;
      ss << "Y wire bounds outside image. z1=" << zcol0 << " z2=" << zcol1 << "."
	 << "min_z=" << meta.min_x() << " max_z=" << meta.max_x() << std::endl;
      throw std::runtime_error( ss.str() );
    }


    // determine range for u-plane
    Double_t xyzStart[3];
    Double_t xyzEnd[3];
#ifdef HAS_LARLITE
    geo->WireEndPoints( 2, zcol0, xyzStart, xyzEnd );
#endif

    float z0 = xyzStart[2];
    Double_t zupt0[3] = { 0,+117.5, z0 };
    int ucol0 = 0;
#ifdef HAS_LARLITE
    geo->NearestWire( zupt0, 0 );
    geo->WireEndPoints( 2, zcol1, xyzStart, xyzEnd );
#endif
    float z1 = xyzStart[2];
    Double_t zupt1[3] = { 0,-117.5, z1-0.1 };
    int ucol1 = 0;
#ifdef HAS_LARLITE
    try {
      ucol1 = geo->NearestWire( zupt1, 0 );
    }
    catch (...) {
      ucol1 = 2399;
    }
#endif

    if ( ucol0>ucol1 ) {
      // this happens on the detector edge
      ucol0 = 0;
    }

    // must fit in _box_pixel_width
    int ddu = ucol1-ucol0;
    int rdu = box_pixel_width%ddu;
    int ndu = ddu/box_pixel_width;
    if ( rdu!= 0) {
      if ( ndu==0 ) {
	// short, extend the end (or lower the start if near end)
	if ( ucol1+rdu<2400 )
	  ucol1+=rdu;
	else
	  ucol0-=rdu;
      }
      else {
	rdu = ddu%box_pixel_width;
	// long, reduce the end
	ucol1 -= rdu;
      }
    }

    // determine v-plane
#ifdef HAS_LARLITE
    geo->WireEndPoints( 2, zcol0, xyzStart, xyzEnd );
#endif
    z0 = xyzStart[2];
    Double_t zvpt0[3] = { 0,-115.5, z0 };
    int vcol0 = 0;
#ifdef HAS_LARLITE
    geo->NearestWire( zvpt0, 1 );
    geo->WireEndPoints( 2, zcol1, xyzStart, xyzEnd );
#endif

    z1 = xyzStart[2];
    Double_t zvpt1[3] = { 0,+117.5, z1-0.1 };
    int vcol1 = 0;
#ifdef HAS_LARLITE
    try {
      vcol1 = geo->NearestWire( zvpt1, 1 );
    }
    catch  (...) {
      vcol1 = 2399;
    }
#endif

    int ddv = vcol1-vcol0;
    int rdv = box_pixel_width%ddv;
    int ndv = ddv/box_pixel_width;
    if ( rdv!= 0) {
      if ( ndv==0 ) {
	// short, extend the end (or lower the start if near end)
	if ( vcol1+rdv<2400 )
	  vcol1+=rdv;
	else
	  vcol0-=rdv;
      }
      else {
	// long, redvce the end
	rdv = ddv%box_pixel_width;
	vcol1 -= rdv;
      }
    }


    // LARCV_DEBUG() << "Pos(Z,T)=(" << zwire << "," << tmid << ") => Crop z=[" << z0 << "," << z1 << "] zcol=[" << zcol0 << "," << zcol1 << "] "
    // 		  << "u=[" << ucol0 << "," << ucol1 << "] du=" << ucol1-ucol0 << " "
    // 		  << "v=[" << vcol0 << "," << vcol1 << "] dv=" << vcol1-vcol0 << " "
    // 		  << "rows=[" << r1 << "," << r2 << "]"
    // 		  << std::endl;

    std::vector<int> crop_coords(8);
    crop_coords[0] = zcol0;
    crop_coords[1] = zcol1;
    crop_coords[2] = ucol0;
    crop_coords[3] = ucol1;
    crop_coords[4] = vcol0;
    crop_coords[5] = vcol1;
    crop_coords[6] = r1;
    crop_coords[7] = r2;

    return crop_coords;

  }

  void UBSplitDetector::clearElapsedTime() {
    elapsed_genbbox = 0.0;
    elapsed_alloc = 0.0;
    elapsed_crop = 0.0;
    num_calls = 0;
  }
  
  void UBSplitDetector::printElapsedTime() {
    LARCV_INFO() << "UBSplitDetector::ElapsedTime ========================" << std::endl;
    LARCV_INFO() << " gen bbox: " << elapsed_genbbox << " secs (ave " << elapsed_genbbox/num_calls << ")" << std::endl;
    LARCV_INFO() << " alloc img: " << elapsed_alloc << " secs (ave " << elapsed_alloc/num_calls << ")" << std::endl;
    LARCV_INFO() << " crop subreg: " << elapsed_crop << " secs (ave " << elapsed_crop/num_calls << ")" << std::endl;
    LARCV_INFO() << "=====================================================" << std::endl;
  }

  void UBSplitDetector::finalize()
  {}

}
#endif
