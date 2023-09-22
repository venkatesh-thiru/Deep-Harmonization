import shapely
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import shape
from shapely import wkt
import rasterio as rio
from rasterio import mask as msk
from rasterio.crs import CRS
from shapely.geometry import box
import h5py
import cv2
import logging
from fiona.drvsupport import supported_drivers

supported_drivers['KML'] = 'rw'


def read_raster(raster_path):
    return rio.open(raster_path, 'r')


def create_buffer(longitude, latitude, buffer, gjson=True):
    '''
    Inputs:
    latitude(epsg=4326)    : Float
    longitude(epsg = 4326) : Float
    buffer(meters)         : Float

    Output : Shapely.Polygon(epsg = 4326)
    '''
    xy = np.array([[longitude, latitude]])
    geoms = [shapely.geometry.Point(pt) for pt in xy]
    geoms_loc = gpd.GeoDataFrame(geoms,
                                 columns=['geometry'])
    geoms_loc = geoms_loc.set_crs(epsg=4326, inplace=True)
    m_coordinate = geoms_loc.to_crs(epsg=32632)
    m_coordinate_buffer = m_coordinate.buffer(buffer).envelope
    geoms_loc.geometry = m_coordinate_buffer.to_crs(epsg=4326)
    if gjson:
        gjson = shapely.geometry.mapping(geoms_loc.geometry.values[0])
        return gjson
    else:
        return geoms_loc.geometry.values[0]


def get_collection_summary(scenes_collection):
    '''
    Inputs:
    scenes_collection    : Descartes Labs Scenes Collection


    Output : Pandas.DataFrame
            Columns(date_acquired, cloud_fraction, sat_id, geometry, coverage, product identifier)
    '''
    df = pd.DataFrame()
    for scene in scenes_collection:

        row = pd.Series({
            'date_acquired': scene.properties.acquired.split('T')[0],
            'cloud_fraction': scene.properties.cloud_fraction,
            'sat_id': scene.properties.sat_id,
            'geometry': scene.geometry,
        })

        if scene.properties.sat_id == 'LANDSAT_8':
            row['identifier'] = scene.properties.identifier
        elif scene.properties.sat_id == 'S2A' or scene.properties.sat_id == 'S2B':
            row['identifier'] = scene.properties.safe_id
        df = df.append(row, ignore_index=True)

    return df


def find_area(row):
    shp_l8 = row['geometry_l8']
    shp_s2 = row['geometry_s2']

    if isinstance(shp_l8, str):
        shp_l8 = wkt.loads(shp_l8)
        shp_s2 = wkt.loads(shp_s2)

    geoms = [shp_l8, shp_s2]

    geoms_loc = gpd.GeoDataFrame(geoms,
                                 columns=['geometry'])

    geoms_loc = geoms_loc.set_crs(epsg=4326, inplace=True)

    m_coordinate = geoms_loc.to_crs(epsg=32632)

    intersect = m_coordinate.geometry.values[0].intersection(m_coordinate.geometry.values[1])

    return intersect.area


def get_item_summary(stac_items):
    df = pd.DataFrame()
    for item in stac_items:
        bounds = item.bbox

        geom = item.geometry
        s = shape(geom)

        row_dict = {
            'date_acquired': item.date,
            'cloud_fraction': item.properties["eo:cloud_cover"] / 100,
            'sat_id': item.properties['platform'],
            'geometry': s.wkt,
            'identifier': item
        }
        if 'landsat:scene_id' in item.properties.keys():
            row_dict['scene_id'] = item.assets['ANG']['href'].split('/')[-2]
        if 's2:datastrip_id' in item.properties.keys():
            row_dict['scene_id'] = item.properties['s2:datastrip_id']

        row = pd.Series(row_dict)
        df = df.append(row, ignore_index=True)

    return df


def mask_raster(raster, mask):
    out_image, out_transform = msk.mask(raster, [mask], crop=True,all_touched = True)

    return out_image, out_transform

def get_bboxs(raster_path,convert_crs = True,target_crs = 4326):
    raster = read_raster(raster_path)
    bounds = raster.bounds
    x_min, y_min, x_max, y_max = bounds
    feature = {
        "type": "Polygon",
    "coordinates": [
       [[x_max, y_min], 
        [x_max, y_max], 
        [x_min, y_max], 
        [x_min, y_min], 
        [x_max, y_min]]
   ]
    }
    if convert_crs:
        raster_crs = raster.crs
        feature_proj = rio.warp.transform_geom(
                        raster_crs,
                        CRS.from_epsg(target_crs),
#                       box(*bounds)
                        feature
                        )
        feature_proj = shape(feature_proj)
        return feature_proj
    else:
        return shape(box(*bounds))
    
def split_bboxes(bbox,patch_size,overlap_size):
    # takes in shapely polygon of intersection
    '''
    example
#     patch_size = 2000,2000
#     image_min = 300000, 5690220
#     image_size = 409800,5800020
#     patch_overlap = 500,500
    '''
    bounds = bbox.bounds
    image_min = int(bounds[0]),int(bounds[1])
    image_size = int(bounds[2]),int(bounds[3])
    zipped = zip(image_min,image_size,patch_size,overlap_size)
    indices = []
    for im_min_dim,im_size_dim,patch_size_dim,patch_overlap_dim in zipped:
        end = im_size_dim + 1 - patch_size_dim
        step = patch_size_dim - patch_overlap_dim
        indices_dim = list(range(im_min_dim,end,step))
        if indices_dim[-1] != im_size_dim - patch_size_dim:
            indices_dim.append(im_size_dim - patch_size_dim)
        indices.append(indices_dim)
    indices_ini = np.array(np.meshgrid(*indices)).reshape(2, -1).T
    indices_ini = np.unique(indices_ini, axis=0)
    indices_fin = indices_ini + np.array(patch_size)
    locations = np.hstack((indices_ini, indices_fin))
    locations = np.array(sorted(locations.tolist()))
    bbox_list = [box(*location) for location in locations]
    return bbox_list

def extract_rgb(array):
    return np.dstack([array[2],array[1],array[0]]) #RGB channels

def stretch_rgb(array):
    '''
    normalize: normalize a numpy array so all value are between 0 and 1
    '''
    normalized = []
    for i in range(3):

        normalized.append(array[:, :, i] * (255.0 / array[:, :, i].max()))
    normalized = np.dstack(normalized)
    return normalized.astype(np.uint8)


def get_tile_midpoint(tile_id,kml_file = "/home/local/DFPOC/thirugv/s2l8h/s2l8h/acquisition_preprocess/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml"):
    gdf = gpd.read_file(kml_file,driver = 'kml')
    row = gdf[gdf.Name == tile_id]
    longitude,latitude = list(row.geometry.values[0].centroid.coords)[0]
    return longitude, latitude


def read_h5file(file_path,key,scale = True):
    try:
        f = h5py.File(file_path)
        array = np.array(f.get(key))
        if scale:
            if key == "s2_image":
                array = array/10000
            elif key == "l8_image" or key == "l8_zoomed":
                array = array * 0.0000275 - 0.2
        return array
    except:
        logging.info(f"error reading patch {file_path}")

def resize_list_of_arrays(stack):
    target_shape = (stack[0].shape[0],stack[0].shape[0])  # shape of first dimension of first array(kind of hard coded)
    resized_stack = [cv2.resize(image,target_shape,interpolation = cv2.INTER_NEAREST) for image in stack]
    return resized_stack
