//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class larcv::Binarize+;
#pragma link C++ class larcv::ADCScaleAna+;
//#pragma link C++ class larcv::ADCScale+;
#pragma link C++ class larcv::CombineImages+;
#pragma link C++ class larcv::SliceImages+;
#pragma link C++ class larcv::Compressor+;
#pragma link C++ class larcv::SegmentAna+;
#pragma link C++ class larcv::StepDigitizer+;
#pragma link C++ class larcv::ADCThreshold+;
#pragma link C++ class larcv::SimpleDigitizer+;
#pragma link C++ class larcv::ResizeImage+;
#pragma link C++ class larcv::CosmicROIFiller+;
#pragma link C++ class larcv::MeanSubtraction+;
#pragma link C++ class larcv::SegmentRelabel+;
#pragma link C++ class larcv::WireMask+;
#pragma link C++ class larcv::EmbedImage+;
#pragma link C++ class larcv::MaskImage+;
#pragma link C++ class larcv::SegmentMask+;
#pragma link C++ class larcv::SegmentMaker+;
#pragma link C++ class larcv::CropROI+;
#pragma link C++ class larcv::ROIMask+;
#pragma link C++ class larcv::ChannelMax+;
#pragma link C++ class larcv::SegWeightTrackShower+;
#pragma link C++ class larcv::ModularPadImage+;
#pragma link C++ class larcv::MultiROICropper+;
#pragma link C++ function larcv::cluster_to_image2d(const larcv::Pixel2DCluster&, size_t, size_t)+;
#pragma link C++ class larcv::LoadImageMod+;
#pragma link C++ class larcv::ROIMerger+;
//ADD_NEW_CLASS ... do not change this line
#endif







