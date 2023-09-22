
import os
import glob
from acquisition_preprocess.acquisition import Acquisition,Acquisition_L2
from acquisition_preprocess.utils import *
from acquisition_preprocess.product_pair import l8s2Pair_L2
import logging
import os
from satsearch import Search
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
# logging.basicConfig(format='%(message)s', level='INFO')

import random
import pystac_client
from odc import stac
import xarray as xr
import pystac
from urllib.parse import urlparse
import rioxarray
import boto3
import warnings

warnings.filterwarnings("ignore")


def odc_stac_base(stac_params):
    stac.configure_s3_access(requester_pays=True)
    stack = stac.load(
                    **stac_params
                    )
    return stack


def stacks_to_gtiff(stack,path,pid):
    tstack = stack.to_array(dim='bands')
    tstack.isel(time=0).rio.to_raster(os.path.join(path,f"{pid}.tif"), overwrite = True)



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

def catalogue_search(buffer, datetime = "2018-01-15/2022-08-10"):
    s2_catalog = "https://earth-search.aws.element84.com/v0"
    s2_collection = "sentinel-s2-l2a-cogs"
    l8_catalog = "https://landsatlook.usgs.gov/stac-server"
    l8sr_collection = "landsat-c2l2-sr"
    l8pan_collection = "landsat-c2l1"

    s2catalog = pystac_client.Client.open(s2_catalog)
    s2items = s2catalog.search(
        intersects = buffer,
        collections = "sentinel-s2-l2a-cogs",
        datetime = datetime,
        query = {"eo:cloud_cover":{"lt":5}}
    ).item_collection_as_dict()


    l8catalog = pystac_client.Client.open(l8_catalog)
    l8items = l8catalog.search(
        intersects = buffer,
        collections = "landsat-c2l2-sr",
        datetime = datetime,
        query = {"eo:cloud_cover":{"lt":5}}
    ).item_collection_as_dict()

    l8pancatalog = pystac_client.Client.open(l8_catalog)
    l8panitems = l8pancatalog.search(
        intersects = buffer,
        collections = "landsat-c2l1",
        datetime = datetime,
        query = {"eo:cloud_cover":{"lt":5}}
    ).item_collection_as_dict()
    if (len(l8items)>0) and (len(s2items)>0) and (len(l8panitems)>0):
        return l8items, s2items, l8panitems
    else:
        return None, None, None

def get_collection(l8items, s2items, l8panitems):
    s2_item_dict = []
    l8_item_dict = []
    l8pan_item_dict = []
    for i,s2_item in enumerate(s2items['features']):
        for j,l8_item in enumerate(l8items['features']):
            if ("properties" in l8_item.keys()) and ("properties" in s2_item.keys()):
                s2_date = s2_item['properties']['datetime'].split('T')[0]
                l8_date = l8_item['properties']['datetime'].split('T')[0]


                if s2_date == l8_date:
                    if (l8_item['properties']['eo:cloud_cover'] < 1) and (s2_item['properties']['eo:cloud_cover'] < 1):
                        if (l8_item['properties']['platform'] == "LANDSAT_9") or (l8_item['properties']['platform'] == "LANDSAT_8"):
                            l8_item_dict.append(l8_item)
                            s2_item_dict.append(s2_item)

                            ppprrr_date = (l8_item['id'].split("_")[2], l8_item['id'].split("_")[3])

                            l8pan_matches = [item for item in l8panitems['features'] if (item['id'].split("_")[2]==ppprrr_date[0]) and (item['id'].split("_")[3]==ppprrr_date[1])]
                            l8pan_item_dict.append(l8pan_matches[0])
                            
    if (len(s2_item_dict) > 0) and (len(l8_item_dict) > 0) and (len(l8pan_item_dict) > 0):
        epsg = s2_item_dict[0]["properties"]["proj:epsg"]
        return pystac.ItemCollection([s2_item_dict[0]]), pystac.ItemCollection([l8_item_dict[0]]), pystac.ItemCollection([l8pan_item_dict[0]]), epsg

    else:
        return None, None, None, None
    
def download_mtl(pan_collection, dir):
    for item in pan_collection:
        uri = item.to_dict()['assets']['MTL.json']['alternate']['s3']['href']
        download_from_s3(uri,dir)
        

def download_collections(s2collections, l8collections, l8pancollections, epsg, geom, patch_dir):
    try:
        bounds = geom.bounds
        s2_stac_params = {
                "items":s2collections,
                "bands" : ["B02", "B03", "B04","B8A","B11","B12","SCL"],
                "bbox" : bounds,
                "resolution" : 10,
                "crs" : epsg,
                "skip_broken_dataset":True,
            }
        s2stack = odc_stac_base(s2_stac_params)
        s2stack.B02.plot.imshow(col="time")
        
        l8_stac_params = {
                "items" : l8collections,
                "bands" : ["blue", "green", "red", 'nir08', 'swir16', 'swir22', "qa_pixel"],
                "resolution":30,
                "bbox":bounds,
                "crs":epsg,
                "patch_url":patch,
                "skip_broken_dataset": True,
            }
        l8stack = odc_stac_base(l8_stac_params)
        
        l8pan_stac_params  = {
                "items" : l8pancollections,
                "bands" : ["pan"],
                "resolution":15,
                "bbox":bounds,
                "crs":epsg,
                "patch_url":patch,
                "skip_broken_dataset": True,
            }
        l8panstack = odc_stac_base(l8pan_stac_params)
        

        download_mtl(l8pancollections, patch_dir)
        stacks_to_gtiff(s2stack, os.path.join(patch_dir), s2collections[0].to_dict()['id'])
        stacks_to_gtiff(l8stack, os.path.join(patch_dir), l8collections[0].to_dict()['id'])
        stacks_to_gtiff(l8panstack, os.path.join(patch_dir), l8pancollections[0].to_dict()['id'])
    except:
        print(f"error-{patch_dir}")


if __name__ == "__main__":
    continents = ["North America"]
    for continent in continents:
        datadir = f"global_test_set/{continent}/"
        gdf = gpd.read_file(f"global_test_set/{continent}.geojson")
        geometries = gdf.geometry.values
        print(f"--------------{continent}------------------")
        for i, geom in enumerate(tqdm(geometries)):
            # geom = geometries[25]
            buffer = geom.buffer(0.05, cap_style=3)
            l8items, s2items, l8panitems = catalogue_search(buffer)
            if (not l8items is None) and (not l8items is None) and (not l8items is None):
                patch_dir = os.path.join(datadir, str(i))
                os.makedirs(patch_dir, exist_ok = True)
                s2collections, l8collections, l8pancollections, epsg = get_collection(l8items, s2items, l8panitems)
                download_collections(s2collections, l8collections, l8pancollections, epsg, buffer, patch_dir)
            






