#RGBViewer
------
Documentation to do...  
.  
.  
.  
.  

###Branches
###roi_feature 
------
documentation for feature branch [LArbys/LArCV/tree/roi_feature](https://github.com/LArbys/LArCV/tree/roi_feature)

also... this is beta... so please contact me (`vgenty@larbys.com`) or open a ticket with issues 

1. Open up RGBViewer in this branch
![alt text](http://www.nevis.columbia.edu/~vgenty/public/_____1.png "1")

2. Toggle the ROITool menu by clicking the **Enable ROITool**
![alt text](http://www.nevis.columbia.edu/~vgenty/public/_____2.png "2")
 
3. If this your first time, or you want to start a new scan specify the output file path and output file ROI producer (this will be the producer name in the LArCV file). I have chosen `output.root` with producer `ahoproducer`. Click **Load Files**.

![alt text](http://www.nevis.columbia.edu/~vgenty/public/_____3.png "3")

4. Lets add an ROI to our image by clicking **Add ROIs**. Something will appear at the bottom left of your screen. There are 3 ROI boxes on top of each other which we will move to the particle location.

![alt text](http://www.nevis.columbia.edu/~vgenty/public/_____4.png "4")

5. I have moved my 3 color boxes to the location of the particles, and resized them to enclose the area I want. **Important note** due to a quirk of `pyqtgraph` once you resize one of the boxes you must make a slight positional adjustment to that box for the other 2 boxes to update in size. I have restricted the size of the boxes to be all the same we can change that if it bothers you. Once you have them in the right spot you must click **Capture ROIs** to save them into memory. To store all ROIs from each of the events into persistent memory in LArCV file on disk click **Store ROIs**.

![alt text](http://www.nevis.columbia.edu/~vgenty/public/_____5.png "5")

6. We can reload the ROI's for a particular event by reloading from the same data file. You have to hit **Replot** to load the ROIs from disk!

![alt text](http://www.nevis.columbia.edu/~vgenty/public/_____6.png "6")


What do all the buttons do?

* **Add ROIs** : adds a set of 3 ROI boxes to the plane for you to move around
* **Remove ROIs** : remove the most recent ROI group (3 rois) from the image
* **Capture ROIs** : store current set of ROIs on the screen into memory
* **Clear ROIs** : clear all the ROIs from the current image
* **Reset ROIs** : clear all ROIs in memory
* **Store ROIs** : store to LArCV file
