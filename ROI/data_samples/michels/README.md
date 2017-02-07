# Michel events

Filtered from EXT-unbiased and MuCS triggers by David Caratelli.

This folder contains the scripts to 
  1. filter the events
  2. make larlite files
  3. run supera to make whole event LArCV files
  4. crop out the images to make HiResDivider images


### Instructions

Many thanks to L. Yates for debugging most of these steps.

  1. First setup a uboonecode environment with the deep learning branch. Setup LiteMaker.
  2. For convenience, make a sim link to it
      
        ln -s [uboonecode folder] larsoft

     The scripts will assume this.
  3. Modify the init.sh script to setup larlite and larsoft environment variables
  4. Modify filter_ext.fcl to point to the michel event list in your folder
  5. Modify the project.py xml file [michel_ext.xml] to point to your data, log, work, software, larsoft directories
  6. Run stage1, which filters the events from the RAW files
  
        project.py --xml michel_ext.xml --stage filter --submit

  7. Wait: check on jobs using
  
        jobsub_q --user [username]

  8. Once jobs done, check them
  
        project.py --xml michel.xml --stage filter --check

  9. If any not good, resubmit by
  
        project.py --xml michel.xml --stage filter --makeup

     Go back to 6.
  10. Once all jobs OK, run LiteMaker
  
        project.py --xml michel.xml --stage larlite --submit

  11. Wait, check. But now check with 
  
        project.py --xml michel.xml --stage larlite --checkana

  12. Locate filesana.list from project.py
  13. Since not too many events, we run LArCV in interactive mode via script:
  
        python run_super.py [location of filesana.list] outdir/

  14. Use hand-scanned BBox file to make HiResDivider file:
  
        command goes here.
