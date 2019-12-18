#include "Utils.h"
namespace larcv{
namespace showermask{
//Note to self.... all of these shamelessly copied from Ralitsa - ask her if stuck
std::vector<double> Utils::GetPosOffsets(std::vector<double> const& point, TH3F* hX, TH3F* hY, TH3F* hZ){

  std::vector<double> offset(3,0.);
  std::vector<double> newpoint;
  if (!IsInsideBoundaries(point) && !IsTooFarFromBoundaries(point)){
    newpoint = PretendAtBoundary(point);
    offset = GetOffsetsVoxel(newpoint,hX,hY,hZ);
  }
  else if(IsInsideBoundaries(point)){
    newpoint = point;
    offset = GetOffsetsVoxel(newpoint,hX,hY,hZ);
  }
  return offset;
}

/// Provides position offsets using voxelized interpolation
std::vector<double> Utils::GetOffsetsVoxel(std::vector<double> const& point,TH3F* hX, TH3F* hY, TH3F* hZ){
    std::vector<double> transformedPoint = Transform(point);
    return {
      hX->Interpolate(transformedPoint[0],transformedPoint[1],transformedPoint[2]),
      hY->Interpolate(transformedPoint[0],transformedPoint[1],transformedPoint[2]),
      hZ->Interpolate(transformedPoint[0],transformedPoint[1],transformedPoint[2])
    };

  }
//----------------------------------------------------------------------------
/// Transform X to SCE X coordinate:  [2.56,0.0] --> [0.0,2.50]
double Utils::TransformX(double xVal){
  double xValNew;
  xValNew = 2.50 - (2.50/2.56)*(xVal/100.0);
  //xValNew -= 1.25;
  return xValNew;
}

//----------------------------------------------------------------------------
/// Transform Y to SCE Y coordinate:  [-1.165,1.165] --> [0.0,2.50]
double Utils::TransformY(double yVal){
  double yValNew;
  yValNew = (2.50/2.33)*((yVal/100.0)+1.165);
  //yValNew -= 1.25;
  return yValNew;
}

//----------------------------------------------------------------------------
/// Transform Z to SCE Z coordinate:  [0.0,10.37] --> [0.0,10.0]
double Utils::TransformZ(double zVal){
  double zValNew;
  zValNew = (10.0/10.37)*(zVal/100.0);

  return zValNew;
}
//----------------------------------------------------------------------------
std::vector<double> Utils::Transform( std::vector<double> const& point){
  return
    { TransformX(point[0]), TransformY(point[1]), TransformZ(point[2]) };
}

//----------------------------------------------------------------------------
/// Check to see if point is inside boundaries of map (allow to go slightly out of range)
bool Utils::IsInsideBoundaries(std::vector<double> const& point){
  return !(
       (point[0] <    0.001) || (point[0] >  255.999)
    || (point[1] < -116.499) || (point[1] > 116.499)
    || (point[2] <    0.001) || (point[2] > 1036.999)
    );
}

bool Utils::InsideFiducial(float x, float y, float z){
  return !(
       (x <    0.001) || (x >  255.999)
    || (y < -116.499) || (y > 116.499)
    || (z <    0.001) || (z > 1036.999)
    );
}

bool Utils::IsTooFarFromBoundaries(std::vector<double> const& point){
  return (
       (point[0] <    0.0) || (point[0] > 266.6)
    || (point[1] < -126.5) || (point[1] > 126.5)
    || (point[2] <    -10) || (point[2] > 1047.0)
    );
}

std::vector<double> Utils::PretendAtBoundary(std::vector<double> const& point){
  double x = point[0], y = point[1], z = point[2];

  if      (point[0] <   0.001)   x =   0.001;
  else if (point[0] >   255.999) x = 255.999;

  if      (point[1] < -116.499) y = -116.499;
  else if (point[1] >  116.499) y =  116.499;

  if      (point[2] <    0.001) z =    0.001;
  else if (point[2] > 1036.999) z = 1036.999;

  return {x,y,z};
}

std::vector<int> Utils::getProjectedPixel( const std::vector<double>& pos3d,
				    const larcv::ImageMeta& meta,
				    const int nplanes,
				    const float fracpixborder ) {
  std::vector<int> img_coords( nplanes+1, -1 );
  float row_border = fabs(fracpixborder)*meta.pixel_height();
  float col_border = fabs(fracpixborder)*meta.pixel_width();

  // tick/row
  float tick = pos3d[0]/(::larutil::LArProperties::GetME()->DriftVelocity()*::larutil::DetectorProperties::GetME()->SamplingRate()*1.0e-3) + 3200.0;
  if ( tick<meta.min_y() ) {
    if ( tick>meta.min_y()-row_border )
      // below min_y-border, out of image
      img_coords[0] = meta.rows()-1; // note that tick axis and row indicies are in inverse order (same order in larcv2)
    else
      // outside of image and border
      img_coords[0] = -1;
  }
  else if ( tick>meta.max_y() ) {
    if ( tick<meta.max_y()+row_border )
      // within upper border
      img_coords[0] = 0;
    else
      // outside of image and border
      img_coords[0] = -1;
  }
  else {
    // within the image
    img_coords[0] = meta.row( tick );
  }

  // Columns
  Double_t xyz[3] = { pos3d[0], pos3d[1], pos3d[2] };
  // there is a corner where the V plane wire number causes an error
  if ( (pos3d[1]>-117.0 && pos3d[1]<-116.0) && pos3d[2]<2.0 ) {
    xyz[1] = -116.0;
  }
  for (int p=0; p<nplanes; p++) {
    float wire = larutil::Geometry::GetME()->WireCoordinate( xyz, p );

    // get image coordinates
    if ( wire<meta.min_x() ) {
      if ( wire>meta.min_x()-col_border ) {
	// within lower border
	img_coords[p+1] = 0;
      }
      else
	img_coords[p+1] = -1;
    }
    else if ( wire>=meta.max_x() ) {
      if ( wire<meta.max_x()+col_border ) {
	// within border
	img_coords[p+1] = meta.cols()-1;
      }
      else
	// outside border
	img_coords[p+1] = -1;
    }
    else
      // inside image
      img_coords[p+1] = meta.col( wire );
  }//end of plane loop

  // there is a corner where the V plane wire number causes an error
  if ( pos3d[1]<-116.3 && pos3d[2]<2.0 && img_coords[1+1]==-1 ) {
    img_coords[1+1] = 0;
  }
  return img_coords;
}

std::vector<double> Utils::MakeSCECorrection (std::vector<double> newvec,
									std::vector<double> original,std::vector<double> offset){
	newvec.resize(3,0);
	newvec[0] = original[0] - offset[0] + 0.6; //mcc9
	newvec[1] = original[1] + offset[1];
	newvec[2] = original[2] + offset[2];
	return newvec;
}

int larcv::showermask::Utils::GetSliceNumber(float resrange){
  int slicenumber;
  if(resrange >= 0 && resrange <= .5) slicenumber = 0;
  if(resrange > .5 && resrange <= 1) slicenumber = 1;
  if(resrange > 1 && resrange <= 1.5) slicenumber = 2;
  if(resrange > 1.5 && resrange <= 2) slicenumber = 3;
  if(resrange > 2 && resrange <= 2.5) slicenumber = 4;
  if(resrange > 2.5 && resrange <= 3) slicenumber = 5;
  if(resrange > 3 && resrange <= 3.5) slicenumber = 6;
  if(resrange > 3.5 && resrange <= 4) slicenumber = 7;
  if(resrange > 4 && resrange <= 4.5) slicenumber = 8;
  if(resrange > 4.5 && resrange <= 5) slicenumber = 9;
  if(resrange > 5 && resrange <= 5.5) slicenumber = 10;
  if(resrange > 5.5 && resrange <= 6) slicenumber = 11;
  if(resrange > 6 && resrange <= 6.5) slicenumber = 12;
  if(resrange > 6.5 && resrange <= 7) slicenumber = 13;
  if(resrange > 7 && resrange <= 7.5) slicenumber = 14;
  if(resrange > 7.5 && resrange <= 8) slicenumber = 15;
  if(resrange > 8 && resrange <= 8.5) slicenumber = 16;
  if(resrange > 8.5 && resrange <= 9) slicenumber = 17;
  if(resrange > 9 && resrange <= 9.5) slicenumber = 18;
  if(resrange > 9.5 && resrange <= 10) slicenumber = 19;

  return slicenumber;
}

float Utils::CalculateSliceWeight(TH1D* muon_slice,TH1D* proton_slice){
  float weight = 1.0;
  int nbins = muon_slice->GetSize();
  float difference = 1000;
  int int_bin;
  float max_muon = 0;
  int max_muon_bin = 0;
  float max_proton = 0;
  int max_proton_bin = 0;
  for (int ii =1;ii<nbins-1;ii++){
    float muonval = muon_slice->GetBinContent(ii);
    float protonval = proton_slice->GetBinContent(ii);
    if (muonval>max_muon){
      max_muon = muonval;
      max_muon_bin = ii;
    }
    if (protonval>max_proton){
      max_proton = protonval;
      max_proton_bin = ii;
    }
  }
  std::cout<<"Max muon value: "<<max_muon <<" @ bin: "<<max_muon_bin<<std::endl;
  std::cout<<"Max Proton value: "<<max_proton <<" @ bin: "<<max_proton_bin<<std::endl;

  int minbin;
  int maxbin;
  if (max_muon_bin > max_proton_bin){
    maxbin = max_muon_bin;
    minbin = max_proton_bin;
  }
  else if (max_proton_bin > max_muon_bin){
    maxbin = max_proton_bin;
    minbin = max_muon_bin;
  }
  else {
    maxbin=max_muon_bin;
    minbin=max_muon_bin;
  }

  // get intersection point
  for(int ii = minbin; ii<=maxbin;ii++){
    if(fabs(muon_slice->GetBinContent(ii) - proton_slice->GetBinContent(ii))<difference){
      difference = fabs(muon_slice->GetBinContent(ii) - proton_slice->GetBinContent(ii));
      int_bin = ii;
    }
  }
  std::cout<< "Intersection @ bin: "<<int_bin<< " with difference: "<< difference <<std::endl;

  float totmuon = muon_slice->GetEntries();
  float muonintersection = 0.0;
  for(int ii = 0; ii<int_bin;ii++){
    muonintersection = muonintersection+muon_slice->GetBinContent(ii);
  }
  std::cout<<"totalmuon entries: "<<totmuon<< " and number before intersection: "<<muonintersection<<std::endl;
  std::cout<< "Makes weight: "<< (muonintersection) <<std::endl;
  weight = muonintersection;
  return weight;
}
}
}
