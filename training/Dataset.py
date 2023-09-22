import torch
import os
import glob
from torch.utils.data import Dataset, DataLoader
from utils import *
import cv2
from models.helpers import perform_band_pass
import logging


class s2l8hDataset(Dataset):
    def __init__(self,csv_path,patch_dir,
                 s2_array_size = 256, l8_array_size = 86,
                 min_field_counts = None,band_pass = True,
                 bandpass_model_path = None, type=None,
                 BOA = True,location = None, date = None):
        '''
        Dataset initialization
        :param csv_path: path to the CSV file containing all patches
        :param patch_dir: cloudless patch directory
        :param min_field_counts: minimum threshold to filter out patches that has no or less number of agricultural fields
        :param band_pass: boolean to apply band-pass function
        :param bandpass_model_path: path to the bandpass transformation model
        :param type: "Train", "Test" or "Validation" patches
        :param BOA: Return BOA reflectance
        :param location: Tile name if tile-specific patches are needed eg:"33UUU"
        :param date: date in "YYYY-mm-dd" format if patches on specific dates are required
        '''
        self.split_df = pd.read_csv(csv_path)
        # Cloud Filter
        self.split_df = self.split_df[(self.split_df['ssim'] >= 0.81) & (self.split_df['psnr'] > 22)] #Commenting this out since these numbers can vary based on the dataset
        if not location is None:
            self.split_df = self.split_df[self.split_df['location'] == location]
        if not date is None:
            self.split_df = self.split_df[self.split_df['date'] == date]
        if not type is None:
            self.split_df = self.split_df[self.split_df['pair_type'] == type]
        if not min_field_counts is None:
            self.split_df = self.split_df[self.split_df['Field_counts'] >= min_field_counts] # The number of fields in a patch is determined using one-soil API the script can be found in extras/onesoil-test-set-filter.ipynb
        self.split_df.reset_index(inplace = True)
        self.patch_dir = patch_dir
        self.BOA = BOA
        self.band_pass = band_pass
        self.bandpass_model_path = bandpass_model_path
        self.l8_size = l8_array_size
        self.s2_size = s2_array_size

    def __len__(self):
        return len(self.split_df)

    def get_by_patchid(self, patch_id):
        row = self.split_df[self.split_df.patch_id == patch_id]
        return self.read_data(row)

    def read_data(self,row):
        patch_id = row.patch_id
        pair_path = row.pair_path
        location = row.location
        date = row.date
        geom = row.geometry

        patch_path = glob.glob(os.path.join(self.patch_dir, "*", "*", f"{patch_id}*"))[0]

        try:
            s2_img = read_h5file(patch_path, "s2_image", self.BOA)
            l8_img = read_h5file(patch_path, "l8_image", self.BOA)

            if self.band_pass:
                l8_img = perform_band_pass(l8_img,model_path = self.bandpass_model_path)

            pan_img = np.expand_dims(read_h5file(patch_path,"l8_pan",scale=False),axis = 0)
            l8_img = cv2.resize(l8_img.transpose(1, 2, 0), [self.l8_size, self.l8_size], interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1) # HARD CODED - HAVE TO CHANGE IN FINAL IMPLEMENTATION
            s2_img = cv2.resize(s2_img.transpose(1, 2, 0), [self.s2_size, self.s2_size], interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
            data_dict = {"pair_path": pair_path, "location": location, "geom": geom, "date": date, "patch_path": patch_path,
                         "s2_img": s2_img.astype('float32').clip(min=0), "l8_img": l8_img.astype('float32').clip(min=0),
                         "l8_pan":pan_img.astype("float32").clip(min = 0)}
        except Exception as e:
            logging.info(f'error reading {patch_path} : {e}')
            return None
        return data_dict

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.split_df.iloc[idx]
        data_dict = self.read_data(row)
        return data_dict

if __name__ == "__main__":


    train_dataset = s2l8hDataset(
        csv_path = "/home/local/DFPOC/thirugv/testing/s2l8h/train_test_validation_patch_example.csv",
        patch_dir = "/home/local/DFPOC/thirugv/testing/s2l8h/DATA_L2_cloudless_patches_example",
        type = "Validation",
        band_pass=False
    )
    print(len(train_dataset))


    # sample_band_pass = test_dataset_band_pass[10]
    # sample = test_dataset[10]
    #
    # l8_rgb = extract_rgb(sample['l8_img'])
    # s2_rgb = extract_rgb(sample_band_pass['l8_img'])
    #
    # # plt.imshow(np.hstack([s2_rgb, cv2.resize(l8_rgb, [86,86], interpolation=cv2.INTER_CUBIC)]))
    # plt.imshow(np.abs(s2_rgb-l8_rgb)[:,:,2],vmin = 0,cmap = 'gray')
    #
    #
    # plt.show()
    # print(len(train_dataset.split_df))

    # test_dataset,validation_dataset = test_dataset[:floor(0.7*len(test_dataset))],test_dataset[floor(0.7*len(test_dataset)):]

    # print(len(train_dataset),len(test_dataset),len(validation_dataset))

    # sample = random.choice(test_dataset)
    # loss = SSIM(data_range=1,channel=6)

    # l8_img,s2_img = torch.from_numpy(sample['l8_img']).unsqueeze(dim=0),torch.from_numpy(sample['s2_img']).unsqueeze(dim=0)
    #
    # l8_zoom = torch.nn.functional.interpolate(l8_img,[256,256])
    # print(loss(s2_img,l8_zoom))

    # print(s2_img.shape,l8_zoom.shape)

    #
    # l8_rgb = extract_rgb(sample['l8_img'])
    # s2_rgb = extract_rgb(sample['s2_img'])


    # plt.imshow(stretch_rgb(np.hstack([s2_rgb,cv2.resize(l8_rgb,[256,256],interpolation = cv2.INTER_NEAREST)])))
    # plt.show()
