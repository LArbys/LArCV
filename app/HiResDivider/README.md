# HiResDivider

Module is used to divide a whole-event image into sub-regions that are based on 3D volumes inside the TPC.

## Parameters

| Parameters | Description |
|------------|:-----------:|
| UseDivFile          | (bool, default true) Load a file that defines the divisions |
| DivisionFile        | (string, only relevant if UseDivFile=True) path to division file |
| NumZdivisions       | (int, only relevant if UseDivFile=False) number of divisions to divide the Z-width of TPC |
| NumYdivisions       | (int, only relevant if UseDivFile=False) number of divisions to divide the Y-width of TPC |
| NumTdivisions       | (int, only relevant if UseDivFile=False) number of divisions to divide the drift dimension of the image |
| OverlapDivisions    | (bool, default false, only relevant if UseDivFile=False) Define overlapping divisions that are offset by half-widths |
| NPlanes             | (int, default 3) Number of planes |
| TickStart           | (int, default 2400) Starting tick of image |
| TickPreCompression  | (int, default 6) Compression of input image in tick-dimension |
| WirePreCompression  | (int, default 1) Compression of input image in wire-dimension |
| MaxWireImageWidth   | (int) Maximum width in wire dimension |
| InputImageProducer  | (string) Producer name for input TPC image |
| InputPMTWeightedProducer    | (string) Producer name for input PMT-weighted image |
| InputROIProducer    | (string) Producer name for input ROI |
| InputSegmentationProducer | (string) Producer name for input segmentation image |
| OutputImageProducer | (string) Producer name for output HiRes divided TPC images |
| OutputROIProducer   | (string) Producer name for output ROIs |
| OutputSegmentationProducer | (string) Producer name for output HiRes divided segmentation images |
| OutputPMTWeightedProducer | (string) Producer name for output HiRes divided PMT-weighted images |
| NumNonVertexDivisionsPerEvent | (int) Number of saved images per event that are blank (no vertex) |
| CropSegmentation | (bool) Perform HiRes division of segmentation image |
| CropPMTWeighted  | (bool) Perform HiRes division of PMT-weighted image |
| NumPixelRedrawThresh   | (list of int) Threshold number of pixels that must be above threshold (InterestingPixelThresh). Threshold for each plane |
| InterestingPixelThresh | (list of float) Pixel intensity threahold in each plane |
| RedrawOnNEmptyPlanes   | (int) Number of planes that must have interesting number of pixels before redrawing |
| MaxRedrawAttempts      | (int) Number of tries to find an interesting division box (with enough pixels above threshold) |
| DivideWholeImage       | (bool) Divide up whole image and save them |


