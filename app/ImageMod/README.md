# ImageMod LArCV Modules

This folder contains a number of different modules that manipulate an LArCV image. Many are used as a preprocessor before the image is passed into caffe or tensorflow.


## Description of each module

Please provide a description of each module here.

### SegmentRelabel

The intention of this module is to relabel the segmentation image.  
For example, we might want to relabel electrons and gammas as "showers" or relabel muon and pions as "tracks".

Parameters

| Parameters | Description |
|------------|:-----------:|
| InputSegmentProducerName | name of input segment image2d |
| OutputSegmentProducerName | name of output  segment image2d. if same as InputSegmentProducerName, then stored in-place |
| LabelMap | PSet connecting new label to a list of old labels. ex. 0:[0,1,2]. You can find example in segmentrelabel.cfg |



