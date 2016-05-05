from .. import larcv

import abc


class PlotImage(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, img_v, roi_v, planes):
        self.imgs = [img_v[i] for i in xrange(img_v.size())]

        if roi_v is not None:
            self.roi_v = [roi_v[i] for i in xrange(roi_v.size())]

        self.planes = planes

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
        self.orig_mat = None
        self.idx = {}

        self.__create_mat__()

        self.plot_mat_t = None

        self.rois = []

        self.reverted = False

    @abc.abstractmethod
    def __create_mat__(self):
        """create load nd_array into 3 channel image"""

    @abc.abstractmethod
    def __threshold_mat__(self, imin, imax):
        """transform threshold orig_mat with imin and imax"""

    @abc.abstractmethod
    def __create_rois__(self):
        """create ROIs meaningfully"""

    @abc.abstractmethod
    def __set_plot_mat__(self):
        """ set how the image will actually be displayed """

    @abc.abstractmethod
    def __revert_image__(self):
        """ revert orig_mat back to it's normal state for caffe"""
        """ example: you have to invert 2nd dimension for pyqt to display"""
        """ the image correctly. So you physically invert it, apply opencv"""
        """ then revert back to send to the network"""

    # void -- threshold original_mat
    def threshold_mat(self, imin, imax):
        self.__threshold_mat__(imin, imax)

    # return the miage that is actually shown
    def set_plot_mat(self):
        return self.__set_plot_mat__()

    # create the ROIs if they exists and return them
    def parse_rois(self):
        self.__create_rois__()
        return self.rois

    # revert the image back to send to caffe!
    def revert_image(self):
        if self.reverted == False:
            self.reverted = True
        else:
            self.reverted = False

        self.__revert_image__()

    # insert thresholded image into self.img_v !
    # since self.orig_mat is a brand new object is this needed? you tell me
    def emplace_image(self):
        for i in self.img_v:
            print "{}".format(i.mean())
        for ch, ix in self.idx.iteritems():
            print ch, ix
            assert self.img_v[ch].shape == self.orig_mat[:, :, ix].shape
            self.img_v[ch] = self.orig_mat[:, :, ix]
