# Code Lifted Wholesale From Shade_States Example
# In Basemap Docs.
# Split to make it possible to save/load cached shapes for faster
# plotting.

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase

# from matplotlib.collections import LineCollection
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")


def create_instance():
    """Make plot using Basemap.  Example is lifted wholesale from
    shade_states example in the Basemap docs.

    """

    # Lambert Conformal map of lower 48 states.

    try:
        m = pickle.load(open("data/usstates.pickle", "rb"))
        m_ = pickle.load(open("data/usstates2.pickle", "rb"))
        # #Need to figure out how to pickle shapeinfo
        # shp_info=pickle.load(open('data/usstates_info.pickle','rb'))
        # shp_info_=pickle.load(open('data/usstates_info2.pickle','rb'))

        print("Loaded Basemap from file")
    except:
        print("Recreated Basemap from scratch")
        m = Basemap(
            llcrnrlon=-119,
            llcrnrlat=20,
            urcrnrlon=-64,
            urcrnrlat=49,
            projection="lcc",
            lat_1=33,
            lat_2=45,
            lon_0=-95,
        )

        # Mercator projection, for Alaska and Hawaii
        m_ = Basemap(
            llcrnrlon=-190,
            llcrnrlat=20,
            urcrnrlon=-143,
            urcrnrlat=46,
            projection="merc",
            lat_ts=20,
        )  # do not change these numbers

        # ---------   draw state boundaries  ----------------------------------------
        ## data from U.S Census Bureau
        ## http://www.census.gov/geo/www/cob/st2000.html
        # shp_info = m.readshapefile(
        #    "data/st99_d00", "states", drawbounds=True, linewidth=0.45, color="gray"
        # )
        # shp_info_ = m_.readshapefile("data/st99_d00", "states", drawbounds=False)
        pickle.dump(m, open("data/usstates.pickle", "wb"), -1)
        pickle.dump(m, open("data/usstates2.pickle", "wb"), -1)
        # #Need to figure out how to picke that shape info, and reload
        # pickle.dump(shp_info,open('data/usstates_info.pickle','wb'),-1)
        # pickle.dump(shp_info_,open('data/usstates_info2.pickle','wb'),-1)
    return m, m_


def make_pop_data():
    ## population density by state from
    ## http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
    popdensity = {
        "New Jersey": 438.00,
        "Rhode Island": 387.35,
        "Massachusetts": 312.68,
        "Connecticut": 271.40,
        "Maryland": 209.23,
        "New York": 155.18,
        "Delaware": 154.87,
        "Florida": 114.43,
        "Ohio": 107.05,
        "Pennsylvania": 105.80,
        "Illinois": 86.27,
        "California": 83.85,
        "Hawaii": 72.83,
        "Virginia": 69.03,
        "Michigan": 67.55,
        "Indiana": 65.46,
        "North Carolina": 63.80,
        "Georgia": 54.59,
        "Tennessee": 53.29,
        "New Hampshire": 53.20,
        "South Carolina": 51.45,
        "Louisiana": 39.61,
        "Kentucky": 39.28,
        "Wisconsin": 38.13,
        "Washington": 34.20,
        "Alabama": 33.84,
        "Missouri": 31.36,
        "Texas": 30.75,
        "West Virginia": 29.00,
        "Vermont": 25.41,
        "Minnesota": 23.86,
        "Mississippi": 23.42,
        "Iowa": 20.22,
        "Arkansas": 19.82,
        "Oklahoma": 19.40,
        "Arizona": 17.43,
        "Colorado": 16.01,
        "Maine": 15.95,
        "Oregon": 13.76,
        "Kansas": 12.69,
        "Utah": 10.50,
        "Nebraska": 8.60,
        "Nevada": 7.03,
        "Idaho": 6.04,
        "New Mexico": 5.79,
        "South Dakota": 3.84,
        "North Dakota": 3.59,
        "Montana": 2.39,
        "Wyoming": 1.96,
        "Alaska": 0.42,
    }

    return popdensity


def plot_us_data(fig, ax, m, m_, data, label_text, title_text):
    # -------- choose a color for each state based on population density. -------
    colors = {}
    statenames = []
    cmap = plt.cm.hot_r  # use 'reversed hot' colormap
    # find max/min.  Then round to 2 dp.

    vmin = round(np.nanmin(data.values), 2)
    vmax = round(np.nanmax(data.values), 2)  # set range.

    norm = Normalize(vmin=vmin, vmax=vmax)
    for shapedict in m.states_info:
        statename = shapedict["NAME"]
        # skip DC and Puerto Rico.
        if statename not in ["District of Columbia", "Puerto Rico"]:
            val = data[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            if np.isnan(val):
                val = vmin
            colors[statename] = cmap(((val - vmin) / (vmax - vmin)))[:3]
        statenames.append(statename)

    # ---------  cycle through state names, color each one.  --------------------
    for nshape, seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ["Puerto Rico", "District of Columbia"]:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor="black", linewidth=0.1)
            ax.add_patch(poly)
            # added to draw outlines.
            # line_collect=LineCollection(seg)
            # ax.add_collection(line_collect)

    # AREA_1 = 0.005  # exclude small Hawaiian islands that are smaller than AREA_1
    # AREA_2 = AREA_1 * 30.0  # exclude Alaskan islands that are smaller than AREA_2
    # AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    # HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    # HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    # AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    # AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    # for nshape, shapedict in enumerate(m_.states_info):  # plot Alaska and Hawaii as map insets
    #     if shapedict['NAME'] in ['Alaska', 'Hawaii']:
    #         seg = m_.states[int(shapedict['SHAPENUM'] - 1)]
    #         if shapedict['NAME'] == 'Hawaii' and float(shapedict['AREA']) > AREA_1:
    #             seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
    #             color = rgb2hex(colors[statenames[nshape]])
    #         elif shapedict['NAME'] == 'Alaska' and float(shapedict['AREA']) > AREA_2:
    #             seg = [(x*AK_SCALE + AK_OFFSET_X, y*AK_SCALE + AK_OFFSET_Y)\
    #                    for x, y in seg]
    #             color = rgb2hex(colors[statenames[nshape]])
    #         poly = Polygon(seg, facecolor=color, edgecolor='gray', linewidth=.45)
    #         ax.add_patch(poly)

    # ---------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8] * 3  # define light gray color RGB
    x1, y1 = m_(
        [-190, -183, -180, -180, -175, -171, -171], [29, 29, 26, 26, 26, 22, 20]
    )
    x2, y2 = m_(
        [-180, -180, -177], [26, 23, 20]
    )  # these numbers are fine-tuned manually
    m_.plot(x1, y1, color=light_gray, linewidth=0.8)  # do not change them drastically
    m_.plot(x2, y2, color=light_gray, linewidth=0.8)

    ax.set_title(title_text)
    # ---------   Show color bar  ---------------------------------------
    ax_c = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    ColorbarBase(ax_c, cmap=cmap, norm=norm, orientation="vertical", label=label_text)
    plt.show()
