from .. import larcv
from .. import np

import abc


class PlotImage(object):
    __metaclass__ = abc.ABCMeta

    # notes
    # work_mat is the ndarray representation of image2d data
    # orig is the 3 channels we will manipulate on the screen
    # plot_mat is modified to have overlays
    # when loading, we manip orig_mat to have the orientation we want
    # before going to caffe, __revert_image__ is called to rearrange the image

    def __init__(self, img_v, roi_v, planes):

        self.imgs = [img_v[i] for i in xrange(img_v.size())]

        if roi_v is not None:
            self.roi_v = [roi_v[i] for i in xrange(roi_v.size())]

        self.planes = None
        self.views  = planes

        ometa = None
        for img in self.imgs:
            if ometa == None:
                ometa = larcv.ImageMeta(img.meta())
            else:
                ometa = ometa.inclusive(img.meta())

        tmp_img_v = []
        for i in xrange(len(self.imgs)):
            meta = larcv.ImageMeta(ometa.width(), ometa.height(), ometa.rows(
            ), ometa.cols(), ometa.min_x(), ometa.max_y(), i)
            img = larcv.Image2D(meta)
            img.paint(0.)
            img.overlay(self.imgs[i])
            tmp_img_v.append(img)

        self.imgs = tmp_img_v
        self.img_v = [larcv.as_ndarray(img) for img in tmp_img_v]

        self.orig_mat = np.zeros(list(self.img_v[0].shape) + [3])
        self.work_mat = np.zeros(list(self.img_v[0].shape) + [len(self.img_v)])

        self.__create_mat()

        self.rois = []

        self.reverted = False

        self.caffe_image = None
        
    def __create_mat(self):

        # load all the images into working matrix
        for ix, img in enumerate(self.img_v):
            self.work_mat[:, :, ix] = self.img_v[ix]

        # put only the selected ones from work_mat into orig_mat
        for p, ch in enumerate(self.views):
            if ch != -1: 
                self.orig_mat[:, :, p] = self.work_mat[:, :, ch]
            else:
                self.orig_mat[:, :, p] = np.zeros((self.orig_mat.shape[0], self.orig_mat.shape[1]))

        self.work_mat = self.work_mat[:,::-1,:]
        self.orig_mat = self.orig_mat[:,::-1,:]

    @abc.abstractmethod
    def __caffe_copy_image__(self):
       """make a copy of work_mat, and do something to it so that it matches"""
       """the way training was done, for most images this means in the subclass"""
       """just return self.work_mat.copy(), in some cases you may want to transpose"""
       """as in the case of the 12 channel image"""


    def set_plot_mat(self,imin,imax):

        self.plot_mat = self.orig_mat.copy()

        # do contrast thresholding
        self.plot_mat[self.plot_mat < imin] = 0
        self.plot_mat[self.plot_mat > imax] = imax

        # make sure pixels do not block each other
        self.plot_mat[:, :, 0][self.plot_mat[:, :, 1] > 0.0] = 0.0
        self.plot_mat[:, :, 0][self.plot_mat[:, :, 2] > 0.0] = 0.0
        self.plot_mat[:, :, 1][self.plot_mat[:, :, 2] > 0.0] = 0.0

        return self.plot_mat

    def __store_orig_mat(self):
         # store the current state of the orig_mat into the working matrix
        for p, ch in enumerate(self.views):
            if ch != -1:
                self.work_mat[:, :, ch] = self.orig_mat[:, :, p]  # don't put a blank in there

    # rswap channels that are shown
    def swap_plot_mat(self,imin,imax,newchs):
        print "swap channels to: ", newchs
        
        self.__store_orig_mat()
        
        # swap the planes
        self.views = newchs
        
        # put work mat values into orig_mat
        for p, ch in enumerate(self.views):
            if ch != -1:
                self.orig_mat[:, :, p] = self.work_mat[:, :, ch]
            else:
                self.orig_mat[:, :, p] = np.zeros((self.orig_mat.shape[0], self.orig_mat.shape[1]))

        # make the viewing plot_mat and return
        return self.set_plot_mat(imin, imax)

    # create the ROIs if they exists and return them
    def parse_rois(self):

        for ix, roi in enumerate(self.roi_v):

            nbb = roi.BB().size()

            if nbb == 0:  # there was no ROI continue...
                continue
            
            r = {}

            r['type'] = roi.Type()
            r['bbox'] = []

            for iy in xrange(nbb):
                bb = roi.BB()[iy]
                r['bbox'].append(bb)

            self.rois.append(r)

        return self.rois

    def __caffe_copy_image(self):
        assert self.reverted == True
        return self.__caffe_copy_image__()

    def revert_image(self):
        self.work_mat = self.work_mat[:,::-1,:]
        self.orig_mat = self.orig_mat[:,::-1,:]
        self.reverted = True

    # insert thresholded image into self.img_v !
    # since self.orig_mat is a brand new object is this needed? you tell me
    def emplace_image(self):

        #original mat may have changed, lets store it 
        self.__store_orig_mat()

        #get the image for caffe
        self.caffe_image = self.__caffe_copy_image()

    def reset_presets(self):
        for key,val in self.presets.iteritems():
            self.preset_layout.removeWidget(val)
            val.setParent(None)
        self.presets = {}
