import numpy as np

def segment_image(ext_proc,pygeo,cosmicid=0):
    img_v=[None,None,None]
    
    for plane in xrange(3):

        trackimg  = ext_proc.TrackImage(plane)
        showerimg = ext_proc.ShowerImage(plane)
        deadimg   = ext_proc.DeadImage(plane)
        thrumuimg = ext_proc.ThruMuImage(plane)
        stopmuimg = ext_proc.StopMuImage(plane)

        #
        # colorize track+shower
        #
        timg = pygeo.image(trackimg)
        simg = pygeo.image(showerimg)

        timg = np.where(timg>10,200,1.0)
        simg = np.where(simg>10,100,1.0)

        img_v[plane] = simg.copy() + timg.copy()
        
        #
        # black out dead image
        #
        
        
        #
        # zero out cosmic image
        #
        tmuimg = pygeo.image(thrumuimg)
        smuimg = pygeo.image(stopmuimg)
    
        smuimg = np.where(smuimg>0.0,-1.0,1.0)
        tmuimg = np.where(tmuimg>0.0,-1.0,1.0)

        if cosmicid == 1:
            img_v[plane] *= tmuimg
        elif cosmicid == 2:
            img_v[plane] *= smuimg
        elif cosmicid == 3:
            img_v[plane] *= tmuimg
            img_v[plane] *= smuimg


        img_v[plane] = img_v[plane].T
        img_v[plane] = img_v[plane][::-1,:]

    return img_v
