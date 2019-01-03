# DLCosmicTag

This module performs vertex reconstruction using LArOpenCV algorithms.

It basically reimplements LArOpenCVHandle but is specific for using the new DLCosmicTag inputs.

### Inputs

* stitched DLCosmicTag larlite file (pixelmask objects) containing ssnet and infill values
* stitched LARCV1 supera file with ADC and CHSTATUS images
* flashmatch+filled pixelmasks and larflow clusters

### Outputs

* larlite particle graph objects
* laropencv particle graph objects
* ntuples for vertex analysis

## Strategies

As a note, we list the strategies we plan to implement and compare.
They go from what is anticipated to be most to least aggressive.
We plan to start with 1 first.

### Strategy 1

We run the vertex code on the individual intime clusters.
This way we enforce the 3D cluster consistency as determine by LArFlow from the get-go.

### Strategy 2

We run the vertex code on a crop around all intime clusters.
We only keep those pixels in the intime-clusters.

Then, we enforce candidate vertices by filtering on 3D consistent clusters.

### Strategy 3

We use the standard vertexing algorithm on an ROI using just the flash position (as the current tagger does),
then reject non-3D consistent clusters.

### Cosmic muon rejection via topology

All strategies will require rejection of vertices on clusters with cosmic-muon topologies.
These will be cuts on

* the end-points vicinity to the edges of the detector,
* straightness
* verticality
* dE/dx arguments (maybe? requires tracking code)

## What's missing

Since we rely on the previous vertex finder, we do not do well on events where
both particles are forward going or back-2-back.
Improving this will require dedicated analyses for the 3D clusters and/or
deep learning classification and/or key-point finding.