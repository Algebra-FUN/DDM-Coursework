import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

regions = {'Hong Kong Island': ['Central and Western',
                                'Wan Chai',
                                'Eastern',
                                'Southern'],
           'Kowloon East': ['Wong Tai Sin',
                            'Kwun Tong'],
           'Kowloon West': ['Yau Tsim Mong',
                            'Sham Shui Po',
                            'Kowloon City'],
           'New Territories East': ['North',
                                    'Tai Po',
                                    'Sha Tin',
                                    'Sai Kung'],
           'New Territories West': ['Kwai Tsing',
                                    'Tsuen Wan',
                                    'Tuen Mun',
                                    'Yuen Long',
                                    'Islands']}
regions_map = {v: k for k, vs in regions.items() for v in vs}
dcca = None

def workshop5(data_path="data/DCCA_21C.json",target_region='New Territories West',author="Your Name"):
    global dcca
    if dcca is None:
        dcca_raw = gpd.read_file(data_path).to_crs(epsg=4326)
        dcca = dcca_raw[['dcca', 'dc_eng', 't_pop', 'geometry']]
        dcca.rename(columns={'dcca': 'id'}, inplace=True)

        dcca['region'] = dcca.dc_eng.apply(lambda k: regions_map[k])
        dcca['density'] = dcca.t_pop/(dcca.to_crs(epsg=6933).area/1e6)/1000

    dcca_selected = dcca[dcca.region == target_region]
    fig,ax = plt.subplots()
    dcca_selected.plot('density', ax=ax, legend=True)
    ax.set_ylabel('population density(1k people per km^2)')
    ax.yaxis.set_label_position("right")
    ax.set_title(author)
    plt.show()

workshop5(author="Your Name")   
