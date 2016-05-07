from plotimage import PlotImage
from .. import np


class DefaultImage(PlotImage):

    def __init__(self, img_v, roi_v, planes):
        super(DefaultImage, self).__init__(img_v, roi_v, planes)
        self.name = "DefaultImage"

    def __create_mat__(self):

        # working copy
        if not hasattr(self, 'work_mat'):
            self.work_mat = np.zeros(list(self.img_v[0].shape) + [len(self.img_v)])

        # compressed images all have the same shape
        self.orig_mat = np.zeros(list(self.img_v[0].shape) + [3])

        for p, fill_ch in enumerate(self.planes):
            self.work_mat[:, :, p] = self.img_v[fill_ch]
            if fill_ch == -1: continue
            self.orig_mat[:, :, p] = self.img_v[fill_ch]
            self.idx[fill_ch] = p

        self.work_mat = self.work_mat[:, ::-1, :]
        self.orig_mat = self.orig_mat[:, ::-1, :]

    def __set_plot_mat__(self, imin, imax):

        self.plot_mat = self.orig_mat.copy()

        # do contrast threhsolding
        self.plot_mat[self.plot_mat < imin] = 0
        self.plot_mat[self.plot_mat > imax] = imax

        # make sure pixels do not block each other
        self.plot_mat[:, :, 0][self.plot_mat[:, :, 1] > 0.0] = 0.0
        self.plot_mat[:, :, 0][self.plot_mat[:, :, 2] > 0.0] = 0.0
        self.plot_mat[:, :, 1][self.plot_mat[:, :, 2] > 0.0] = 0.0

        return self.plot_mat


    def __swap_mat_channels__(self, imin, imax, newchs):
        print "swap channels to: ", newchs
        # store the current state of the orig_mat into the working matrix
        for p, ch in enumerate(self.planes):
            if ch != -1:
                self.work_mat[:, :, ch] = self.orig_mat[:, :, p]  # don't put a blank in there
        # swap the planes
        self.planes = newchs
        # put work mat values into orig_mat
        for p, ch in enumerate(self.planes):
            if ch != -1:
                self.orig_mat[:, :, p] = self.work_mat[:, :, ch]
            else:
                self.orig_mat[:, :, p] = np.zeros((self.orig_mat.shape[0], self.orig_mat.shape[1]))
        # make the viewing plot_mat and return
        return self.__set_plot_mat__(imin, imax)

    # revert back to how image was in ROOTFILE
    def __revert_image__(self):
        self.orig_mat = self.orig_mat[:, ::-1, :]
        self.work_mat = self.work_mat[:, ::-1, :]

    def __create_rois__(self):

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
