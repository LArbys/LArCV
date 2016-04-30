import caffe

import displayfetcher as df

class RGBDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        self._params = df['net_params']
        self._img    = df['rgb_params']
        
        data_shape = (1,
                      self._params['channels'],
                      self._img['width'],
                      self._img['height'])
        
        #label_shape = (1,)

        top[0].reshape(*data_shape)
        #top[1].reshape(*label_shape)


    def reshape(self,bottom,top):
        pass

    def forward(self,bottom,top):
        blob    = df.fetch_blob()

        # Copy data into net's input blobs
        top[0].data[...] = blob.astype(np.float32, copy=False)  
        

    def backward(self,top,propogate_down,bottom):
        pass

    def reshape(self,bottom,top):
        pass
        
