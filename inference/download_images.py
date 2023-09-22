import pystac_client
import geopandas as gpd
from shapely.geometry import shape,Point
import numpy as np
import rioxarray
from tqdm import tqdm
from odc import stac
import xarray as xr
import pystac
import os
import dask
from concurrent.futures import ThreadPoolExecutor
import urllib,json
from urllib.parse import urlparse
import boto3
import logging
import glob
import rasterio as rio
from datetime import datetime
import pandas as pd

stac.configure_s3_access(requester_pays = True)

def patch(uri):
    return uri.replace('https://landsatlook.usgs.gov/data/', 's3://usgs-landsat/')

# SOME HELPERS TO DOWNLOAD DATA FROM AWS S3 REQUESTER PAYS BUCKET
def split_s3_path(s3_path): # RS Data Services
    """Splits a S3 path into bucket and prefix.

    Args:
        s3_path: Full s3 path.

    Returns:
        bucket name and prefix
    """
    s3parse = urlparse(s3_path, allow_fragments=False)
    bucket = s3parse.netloc
    s3_prefix = s3parse.path
    s3_prefix = s3_prefix.strip("/")

    return bucket, s3_prefix


def download_from_s3(url, target_dir):
    identifier = url.split('/')[-2]
    bucket, prefix = split_s3_path(url)

    s3_client = boto3.Session().client('s3')
    response = s3_client.get_object(Bucket=bucket,
                                    Key=prefix,
                                    RequestPayer='requester')
    content = response['Body'].read()
    with open(os.path.join(target_dir, f"{identifier}.json"), 'wb') as file:
        file.write(content)

def stacks_to_gtiff(stack,path,pids):
    tstack = stack.to_array(dim='bands')
    for i in tqdm(range(0,tstack.shape[1])):
        tstack.isel(time = i).rio.to_raster(os.path.join(path,f"{pids[i]}.tif"), overwrite = True)


def extract_date(files, source):
    mapping = {}
    for file in files:
        file = os.path.split(file)[-1]
        if source == "S2":
            date = file.split('_')[2].split('T')[0]
        if source == "L8":
            date = file.split('_')[3]
        date = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"

        mapping[date] = file
    return mapping


def get_cloud_mask_l8(array, flags_list=['fill', 'cirrus', 'cloud', 'dilated_cloud', 'shadow']):
    l8_mask_flags = {
        'fill': 1 << 0,
        'dilated_cloud': 1 << 1,
        'cirrus': 1 << 2,
        'cloud': 1 << 3,
        'shadow': 1 << 4,
        'snow': 1 << 5,
        'clear': 1 << 6,
        'water': 1 << 7
    }
    final_mask = np.zeros_like(array)
    for flag in flags_list:
        flag_mask = np.bitwise_and(array, l8_mask_flags[flag])
        final_mask = final_mask | flag_mask

    return np.squeeze(final_mask > 0)


def get_cloud_mask_s2(array, flags_list=['NO_DATA','CLOUD_SHADOWS', 'THIN_CIRRUS', 'CLOUD_MEDIUM_PROBABILITY',
                                         'CLOUD_HIGH_PROBABILITY']):
    scl_keys = {
        'NO_DATA': 0,
        'SATURATED_OR_DEFECTIVE': 1,
        'DARK_AREA_PIXELS': 2,
        'CLOUD_SHADOWS': 3,
        'VEGETATION': 4,
        'NOT_VEGETATED': 5,
        'WATER': 6,
        'UNCLASSIFIED': 7,
        'CLOUD_MEDIUM_PROBABILITY': 8,
        'CLOUD_HIGH_PROBABILITY': 9,
        'THIN_CIRRUS': 10,
        'SNOW': 11,
    }
    final_mask = np.zeros_like(array)
    for flag in flags_list:
        flag_mask = (array == scl_keys[flag])
        final_mask = final_mask + flag_mask

    return np.squeeze(final_mask > 0)


def calculate_valid_proportion(file, directory, mask_generator):
    if not file is np.nan:
        file_path = os.path.join(directory, file)
        raster = rio.open(file_path, 'r')
        array = raster.read()
        mask = array[-1].astype(int)
        mask = mask_generator(mask)
        return np.count_nonzero(mask) / mask.size
    else:
        return np.nan

def to_square(geom,buffer):
    minx, miny, maxx, maxy = geom.bounds
    centroid = [(maxx+minx)/2, (maxy+miny)/2]
    return Point(centroid).buffer(buffer, cap_style=3)

class acquire_data:
    def __init__(self, geom_file, name, data_dir, date_range,buffer = None):

        self._geom_file = geom_file
        self._name = name
        self._data_dir = data_dir
        self._date_range = date_range

        os.makedirs(os.path.join(self._data_dir,self._name,"DN","S2"),exist_ok=True)
        os.makedirs(os.path.join(self._data_dir,self._name,"DN","L8"),exist_ok=True)
        os.makedirs(os.path.join(self._data_dir,self._name,"DN","L8_pan","mtl"),exist_ok=True)

        # PREDEFINED COLLECTIONS
        self.s2_catalog = "https://earth-search.aws.element84.com/v1"
        self.s2_collection = "sentinel-s2-l2a-cogs"
        self.l8_catalog = "https://landsatlook.usgs.gov/stac-server"
        self.l8sr_collection = "landsat-c2l2-sr"
        self.l8pan_collection = "landsat-c2l1"
        # PLACE HOLDER
        self.target_epsg = None
        self.ppprrr_dates = None
        self._exe = ThreadPoolExecutor(max_workers=5)

        gdf = gpd.read_file(self._geom_file)
        gdf = gdf.set_crs(epsg=4326)
        self.geom = gdf.geometry[0]
        if buffer is not None:
            self.geom = to_square(self.geom,buffer)

        self.bounds = self.geom.bounds

    def odc_stac_base(self, stac_params):
        stac.configure_s3_access(requester_pays=True)
        stack = stac.load(
                        **stac_params
                        )
        return stack

    def download_s2(self):
        catalog = pystac_client.Client.open(self.s2_catalog)
        items = catalog.search(
            intersects=self.geom,
            collections = self.s2_collection,
            datetime = self._date_range,
            query = {"eo:cloud_cover" : {"lt":5}}
            ).item_collection()
        self.target_epsg = items[0].to_dict()['properties']['proj:epsg']
        pids = [item.to_dict()['properties']['sentinel:product_id'] for item in items]
        exe = ThreadPoolExecutor(max_workers=5)

        stac_params = {
            "items":items,
            "bands" : ["B02", "B03", "B04","B8A","B11","B12","SCL"],
            "bbox" : self.bounds,
            "resolution" : 10,
            "crs" : self.target_epsg,
            "progress" : tqdm,
            "pool" : exe,
            "skip_broken_dataset":True,
        }
        stack = self.odc_stac_base(stac_params)
        stacks_to_gtiff(stack,os.path.join(self._data_dir,self._name,"DN","S2"),pids)

    def download_l8(self):
        catalog = pystac_client.Client.open(self.l8_catalog)
        items = catalog.search(
            intersects=self.geom,
            collections=self.l8sr_collection,
            datetime=self._date_range,
            query={"eo:cloud_cover": {"lt": 10}}
        ).item_collection()
        items_dict = items.to_dict()['features']
        items_dict_l8 = [item for item in items_dict if item['properties']['platform'] == "LANDSAT_8"]
        pids = [str(item['id']) for item in items_dict_l8]
        self.ppprrr_date = [(str(item['id'].split('_')[2]), str(item['id'].split('_')[3])) for item in items_dict_l8]
        item_collection = pystac.ItemCollection(items_dict_l8)
        exe = ThreadPoolExecutor(max_workers=5)
        stac_params  = {
                        "items" : item_collection,
                        "bands" : ["blue", "green", "red", 'nir08', 'swir16', 'swir22', "qa_pixel"],
                        "resolution":30,
                        "bbox":self.bounds,
                        "crs":self.target_epsg,
                        "progress":tqdm,
                        "patch_url":patch,
                        "pool": exe,
                        "skip_broken_dataset": True,
                        }
        stack = self.odc_stac_base(stac_params)
        stacks_to_gtiff(stack, os.path.join(self._data_dir, self._name, "DN", "L8"),pids)

    def download_l8pan(self):
        catalog = pystac_client.Client.open(self.l8_catalog)
        items = catalog.search(
            intersects=self.geom,
            collections=self.l8pan_collection,
            datetime=self._date_range,
            query={"eo:cloud_cover": {"lt": 10}}
        ).item_collection()
        items_dict = items.to_dict()['features']
        items_dict_l8 = [item for item in items_dict if item['properties']['platform'] == "LANDSAT_8"]
        ppprrr_date_mapping = {
            (str(items_dict_l8[i]['id'].split('_')[2]), str(items_dict_l8[i]['id'].split('_')[3])): i for i in
            range(0, len(items_dict_l8))
        }
        matching_dates = [ppprrr_date_mapping[tup] for tup in self.ppprrr_date]
        items_dict_match = [items_dict_l8[i] for i in matching_dates]
        pids = [str(item['id']) for item in items_dict_match]
        item_collection = pystac.ItemCollection(items_dict_match)
        exe = ThreadPoolExecutor(max_workers=5)
        stac_params  = {
                        "items" : item_collection,
                        "bands" : ["pan"],
                        "resolution":15,
                        "bbox":self.bounds,
                        "crs":self.target_epsg,
                        "progress":tqdm,
                        "patch_url":patch,
                        "skip_broken_dataset": True,
                        "pool":exe,
                        }
        stack = self.odc_stac_base(stac_params)
        stacks_to_gtiff(stack, os.path.join(self._data_dir, self._name, "DN", "L8_pan"),pids)
        self.download_mtl(item_collection)

    def download_mtl(self, pan_collection):
        for item in tqdm(pan_collection):
            uri = item.to_dict()['assets']['MTL.json']['alternate']['s3']['href']
            download_from_s3(uri,os.path.join(self._data_dir, self._name, "DN", "L8_pan","mtl"))

    def write_metadata(self):

        # datadir = "DATADIR/California"
        raster_path = os.path.join(self._data_dir,self._name, "DN")
        s2_files = glob.glob(os.path.join(raster_path, 'S2', '*.tif'))
        l8_files = glob.glob(os.path.join(raster_path, 'L8', '*.tif'))
        pan_files = glob.glob(os.path.join(raster_path, 'L8_pan', '*.tif'))

        mapping_s2 = extract_date(s2_files, "S2")
        mapping_l8 = extract_date(l8_files, "L8")
        mapping_pan = extract_date(pan_files, "L8")

        all_dates = list(mapping_s2.keys()) + list(mapping_l8.keys())
        all_dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

        df = pd.DataFrame(all_dates, columns=['dates'])
        df['S2'] = df['dates'].map(mapping_s2)
        df['L8MS'] = df['dates'].map(mapping_l8)
        df['L8PAN'] = df['dates'].map(mapping_pan)

        df['L8_invalid'] = df['L8MS'].apply(calculate_valid_proportion,
                                            args=(os.path.join(self._data_dir,self._name,"DN","L8"), get_cloud_mask_l8))
        df['S2_invalid'] = df['S2'].apply(calculate_valid_proportion, args=(os.path.join(self._data_dir,self._name,"DN","S2"), get_cloud_mask_s2))

        df.to_csv(os.path.join(self._data_dir,self._name, "image_mapping.csv"), index=False)


if __name__ == "__main__":
    case = acquire_data("geoms/Frenso/Frenso_buffer.geojson", "Frenso", "DATADIR","2021-09-20/2022-09-20")
    case.download_s2()
    case.download_l8()
    case.download_l8pan()
    case.write_metadata()
