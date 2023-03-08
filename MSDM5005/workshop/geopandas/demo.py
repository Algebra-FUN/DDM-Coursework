import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

def coords(target):
    return np.array([target.centroid.x.item(), target.centroid.y.item()])

def gpd_workshop(origin,target,author,xlims,ylims):
    Target = world[world.name == target["name"]]
    Origin = cities[cities.name == origin["name"]]
    ax = plt.gca()
    world_union = gpd.GeoSeries(world.unary_union)
    world_union.plot(ax=ax,color="#fefee5")
    world_union.boundary.plot(ax=ax,color='k',lw=.5)
    Target.plot(ax=ax,color=target["color"])
    Origin.plot(ax=ax,color=origin["color"],marker=origin["marker"])
    ax.set_facecolor("#c9ecff")
    plt.arrow(*coords(Origin),*(coords(Target)-coords(Origin)),head_width = 1.5,width = 0.5,color="orange")
    plt.xlim(*xlims)
    plt.ylim(*ylims)
    plt.scatter([],[],marker=origin["marker"],color=origin["color"],label=f"from {origin['name']}")
    plt.scatter([],[],marker=target["marker"],color=target["color"],label=f"to {target['name']}")
    plt.legend(loc="lower center",ncol=2,bbox_to_anchor=(.5,-.3),frameon=False)
    plt.title(author)
    plt.show()

target = {
    "name":"Ukraine",
    "color":"green",
    "marker":"o"
}
origin = {
    "name":"Shanghai",
    "color":"red",
    "marker":"*"
}

gpd_workshop(origin,target,author="Your NAME",xlims=(0,130),ylims=(10,60))