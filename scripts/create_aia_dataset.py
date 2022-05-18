import numpy as np
from tqdm import tqdm
import os
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import rgb2gray


def get_multiple_year_image(image_type):
    """
        concatenate data of multiple years [image]
    """
    year_dict = {"start" : 2010, "end" : 2017}
    path = "./datasets/aia1600/"

    for i, year in enumerate(tqdm(range(year_dict["start"],
                                year_dict["end"]+1))):
        data_path_256 = path + str(year) + "_" + image_type + "_512.npy"
        data_path_512 = path + str(year) + ".npy"
        # image_data = np.load(data_path_512)
        # print(np.max(image_data[0,0,:,:]))
        # print(np.max(resize(image_data[0,0,:,:],(256,256))))

        if not os.path.exists(data_path_256):
            image_data = np.load(data_path_512) # (B, C, H, W) uint8
            N,C,H,W = image_data.shape
            # _image_data = np.empty((N,1,512,512)) # float64
            _image_data = np.empty((N,512,512)).astype(np.uint8)
            for n in range(N):
                source = image_data[n,:,:,:].astype(np.uint8)
                source = source.transpose(1,2,0) # (H, W, C) uint8
                # print(np.max(source))
                # print(np.max(resize(source,(256,256))))
                if C == 3:
                    source = img_as_ubyte(rgb2gray(source)) # (H , W) float64
                    # source = source[:,:,np.newaxis]

                # todo:np.uint8だとresize時に0-1で正規化されるっぽい→要検証
                # _image_data[n,:,:,:] = resize(source,(512,512)).transpose(2,0,1) 
                _image_data[n,:,:] = source
            
            image_data = _image_data
            np.save(data_path_256,image_data)
        else:
            image_data = np.load(data_path_256)