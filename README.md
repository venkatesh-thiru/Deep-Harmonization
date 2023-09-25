# Deep-Harmonization
Code for "Deep Learning-based Harmonization and Super-Resolution of Landsat-8 and Sentinel-2 images" paper.

![s2l8h](images/comparison.png)

## Table of Content
* [Abstract](#abstract)
* [Installation](#installation)

## Abstract
Multi-spectral satellite images from the Earth’s surface are used in various applications spanning from water quality assessment and urban planning to climate monitoring, disaster response, infrastructure oversight, and agricultural surveillance. 
Many of these applications would benefit from higher spatial and temporal resolution of observations, which could be achieved by
combining observations from several different sources. 
In this study, we introduce a deep learning-based pipeline to harmonize the spectral and spatial discrepancies between the Landsat-8 and Sentinel-2 Earth Observation satellites. 
Through established image quality metrics, we demonstrate a significant enhancement in the spatial resolution of Landsat-8 images. 
Field studies show that leveraging unified images from both satellites increases the availability of cloud-free images by up to 60% annually in our area of study. 
Additionally, our pipeline enhances the Normalized Difference Vegetation Index (NDVI) correlation between Landsat-8 and Sentinel-2 observations by an absolute 4.5%, while also offering significant performance gains in a downstream crop segmentation task. 
We demonstrate that our 100M parameter model, which is trained on data from Europe, generalizes to most regions of the world, with only minor limitations. 
Furthermore, we show that the pipeline is able to provide uncertainty estimates for its outputs, which are valuable for decision-making in downstream applications.

## Installation
The [environment.yml](environment.yml) file can be used to clone the conda environment used for the development of this project.
Follow the instructions given below.

```
>> conda env create --file environment.yml
>> conda activate s2l8h
```
The given environment file consists of libraries that are used to process raw rasters, download data from STAC compliant APIs and other visualization libraries, Which are necessary to run the [inference](inference/inference_example.ipynb) examples provided.
It is recommended to build the environment from the given file as there are know inconsistency issues with some source dependency installations from pip.

The project uses the following libraries:

[Pytorch](https://pytorch.org/get-started/locally/) -  Deep Learning model training and inferencing.

[transformers](https://huggingface.co/docs/transformers/installation) - To access the models using hugging face model card.

[YACS](https://github.com/rbgirshick/yacs) - Configuration management.

[ODC-STAC](https://github.com/opendatacube/odc-stac) - access and load Geo Spatial data from STAC compliant APIs.

[Geopandas](https://geopandas.org/en/stable/getting_started/install.html) - Geo Spatial data manipulation.

[Rasterio](https://rasterio.readthedocs.io/en/stable/) -  Raster data handling.

[h5py](https://pypi.org/project/h5py/) -  Dataset file format.


**NOTE** : [BOTO3](https://pypi.org/project/boto3/) is also required to access data from S3 buckets, please set things up as described in the instructions page [here](https://pypi.org/project/boto3/).

## Usage
### Dataset
The models were trained-tested and validated on about 60K Sentinal-2 and Landsat-8 image pairs, the bounds for each pairs can be found in the [meta-data file](training/train_test_validation_patch_extended_final.csv).


### Training
Training configurations are defined in the [config](training/config.py) file.
