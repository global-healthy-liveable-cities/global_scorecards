# scorecard functions
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from textwrap import wrap
from fpdf import FPDF, FlexTemplate

# switch to low res for quick prototyping
debug = False

def fpdf2_mm_scale(mm):
    # returns a width double that of the conversion of mm to inches
    # because that seems to work about right, based on trial and error
    return(2*mm/25.4)


## radar chart
def li_profile(city_stats, comparisons,title,cmap,path, width = fpdf2_mm_scale(80), height = fpdf2_mm_scale(80), dpi = 300):
    """
    Generates a radar chart for city liveability profiles
    Drawing on https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib
    
    Arguments:
        city_stats  A pandas series of indicators for a particular city
        comparisons A dictionary of pandas series IQR point summaries (p25, p50 and p75)
        cmap A colour map
    """
    figsize=(width, height)
    # Values for the x axis
    ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(city_stats), endpoint=False)
    VALUES = city_stats.values
    COMPARISON = comparisons['p50'].values
    INDICATORS = city_stats.index
    # Colours
    GREY12 = "#1f1f1f"
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    COLORS = cmap(list(norm(VALUES)))
    # Initialize layout in polar coordinates
    textsize = 12
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    # Set background color to white, both axis and figure.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(1.2 * np.pi / 2)
    ax.set_ylim(-50, 125)
    # Add geometries to the plot -------------------------------------
    # Add bars to represent the cumulative track lengths
    ax.bar(ANGLES, VALUES, color=COLORS, alpha=0.9, width=0.52, zorder=10)
    # Add interquartile comparison reference lines
    ax.vlines(ANGLES, 
              comparisons['p25'], 
              comparisons['p75'],
              color=GREY12, 
              zorder=11)
    # Add dots to represent the mean gain
    ax.scatter(ANGLES, COMPARISON, s=60, color=GREY12, zorder=11)
    # Add labels for the regions
    LABELS = ["\n".join(wrap(r, 10, break_long_words=False)) for r in INDICATORS]
    # Set the labels
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(LABELS, size=textsize);
    # Remove lines for polar axis (x)
    ax.xaxis.grid(False)
    # Put grid lines for radial axis (y) at 0, 1000, 2000, and 3000
    ax.set_yticklabels([])
    ax.set_yticks([0, 25, 50, 75, 100])
    # Remove spines
    ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    # Adjust padding of the x axis labels ----------------------------
    # This is going to add extra space around the labels for the 
    # ticks of the x axis.
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(10)
    # Add custom annotations -----------------------------------------
    # The following represent the heights in the values of the y axis
    PAD = -2.3
    for num in [0,50,100]:
        ax.text(-0.2 * np.pi / 2, num + PAD, f"{num}", ha="center", size=textsize)
    # Add text to explain the meaning of the height of the bar and the
    # height of the dot
    ax.text(ANGLES[0], -50, title, rotation=0, 
            ha="center", va="center", size=textsize, zorder=12)
    ax.text(ANGLES[0]+ 0.012, 
            comparisons['p50'][0] + 10, 
            "\n\nGlobal comparison", rotation=0, 
            ha="right", va="center", size=0.9*textsize, zorder=12)
    fig.savefig(path,dpi=dpi) 
    plt.close(fig)    

## Spatial distribution mapping       
def spatial_dist_map(gdf,
                     column,
                     range,
                     label,
                     tick_labels,
                     cmap,
                     path,
                     width = fpdf2_mm_scale(88), 
                     height = fpdf2_mm_scale(80),
                     dpi = 300):
    """
    Spatial distribution maps using geopandas geodataframe
    """
    figsize=(width, height)
    textsize = 14
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off();
    divider = make_axes_locatable(ax)  # Define 'divider' for the axes
    # Legend axes will be located at the 'bottom' of figure, with width '5%' of ax and
    # a padding between them equal to '0.1' inches 
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    gdf.plot(column=column,
            ax=ax,
            legend=True,
            vmin = range[0],
            vmax = range[1],
            legend_kwds={'label':label,
                         'orientation':'horizontal'},
            cax=cax,
            cmap = cmap)
    scalebar = AnchoredSizeBar(ax.transData,
                               20000, '20 km', 'upper left', 
                               pad=0,
                               color='black',
                               frameon=False,
                               size_vertical=2,
                               fontproperties=fm.FontProperties(size=textsize))
    ax.add_artist(scalebar)
    x, y, arrow_length = 1, 1, 0.06
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=4, headwidth=8),
                ha='center', va='center', fontsize=textsize,
                xycoords=ax.transAxes)
    cax.tick_params(labelsize=textsize)
    cax.xaxis.label.set_size(textsize)
    if tick_labels != None:
        #cax.set_xticks(cax.get_xticks().tolist())
        #cax.set_xticklabels(tick_labels)
        cax.xaxis.set_major_locator(ticker.MaxNLocator(len(tick_labels)))
        ticks_loc = cax.get_xticks().tolist()
        cax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        cax.set_xticklabels(tick_labels)
    plt.tight_layout()
    fig.savefig(path,dpi=dpi) 
    plt.close(fig)   

def threshold_map(gdf, column, range,scale,comparison, label,cmap,path,width = fpdf2_mm_scale(88),height = fpdf2_mm_scale(80), dpi = 300):
    figsize=(width, height)
    textsize = 14
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off();
    divider = make_axes_locatable(ax)  # Define 'divider' for the axes
    # Legend axes will be located at the 'bottom' of figure, with width '5%' of ax and
    # a padding between them equal to '0.1' inches 
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    gdf.plot(column=column,
            ax=ax,
            legend=True,
            #vmin = range[0],
            #vmax = range[1],
            legend_kwds={'label':label,
                         'orientation':'horizontal'},
            cax=cax,
            cmap=cmap,
            #norm = mpl.colors.LogNorm(vmin=1, vmax=range[1])
            )
    scalebar = AnchoredSizeBar(ax.transData,
                               20000, '20 km', 'upper left', 
                               pad=0,
                               color='black',
                               frameon=False,
                               size_vertical=2,
                               fontproperties=fm.FontProperties(size=textsize))
    ax.add_artist(scalebar)
    x, y, arrow_length = 1, 1, 0.06
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=4, headwidth=8),
                ha='center', va='center', fontsize=textsize,
                xycoords=ax.transAxes)
    # Add threshold band on legend
    extrema = cax.get_xlim()
    ax_low  = (comparison[0]-extrema[0])/(extrema[1]-extrema[0])
    ax_high = (comparison[1]-extrema[0])/(extrema[1]-extrema[0])
    ax_diff = ax_high-ax_low
    cax.annotate('', 
                 xy=(ax_low, 0.5), 
                 xytext=(ax_high,0.5),
                arrowprops=dict(facecolor='black', arrowstyle="-"),
                ha='center', va='center', fontsize=textsize,
                xycoords=cax.transAxes)
    cax.xaxis.set_major_formatter(ticker.EngFormatter())
    cax.tick_params(labelsize=textsize)
    cax.xaxis.label.set_size(textsize)
    plt.tight_layout()
    fig.savefig(path,dpi=dpi) 
    plt.close(fig) 

def setup_thresholds(csv_thresholds_data,threshold_lookup):
    thresholds = pd.read_csv(csv_thresholds_data)
    cities = thresholds.columns[1:]
    # Threshold analysis
    threshold_index_cols = ['scenario','description','location']
    thresholds['scenario'] = thresholds.iloc[:,0].apply(lambda x: x.split(' - ')[0].split('_')[-1])
    thresholds['description'] = thresholds.iloc[:,0].apply(lambda x: x.split(' - ')[1])
    thresholds['location'] = thresholds.iloc[:,0].apply(lambda x: x.split(' - ')[2])
    thresholds = thresholds[threshold_index_cols+list(cities)]
    # Extract threshold scenarios from variables
    threshold_scenarios = thresholds.loc[thresholds.location.str.startswith('within')]
    threshold_scenarios = threshold_scenarios.loc[threshold_scenarios.scenario=='B']
    threshold_scenarios['lower'] = threshold_scenarios.location.apply(lambda x: int(x[0:-1].split('(')[1].split(', ')[0]))
    threshold_scenarios['upper'] = threshold_scenarios.location.apply(lambda x: int(x[0:-1].split('(')[1].split(', ')[1]))
    threshold_scenarios = (threshold_scenarios.set_index('description'))
    threshold_scenarios = {
        'data' : threshold_scenarios,
        'lookup' : threshold_lookup
    }
    return threshold_scenarios


def policy_rating(range,
                  score,
                  cmap, 
                  comparison = None,
                  width = fpdf2_mm_scale(70), 
                  height = fpdf2_mm_scale(15),
                  label = 'Policies identified',
                  path='policy_rating_test.jpg',
                  dpi = 300):
    textsize = 14
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(bottom=0)
    cmap = cmap
    norm = mpl.colors.Normalize(vmin=range[0], vmax=range[1])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, 
        orientation='horizontal',
         #shrink=0.9, pad=0, aspect=90
        )
    # Format Global ticks
    if comparison is None:
        ax.xaxis.set_ticks([])
    else:
        ax.xaxis.set_major_locator(ticker.FixedLocator([comparison['50%']]))
        ax.set_xticklabels([f'25 city median'])
        ax.tick_params(labelsize=textsize)
    # Format City ticks
    ax_city = ax.twiny()
    ax_city.set_xlim(range)
    ax_city.xaxis.set_major_locator(ticker.FixedLocator([score]))
    sep = ''
    if comparison is not None and label=='':
        sep = '\n'
    ax_city.set_xticklabels([f"{sep}{str(score).rstrip('0').rstrip('.')}/{range[1]}{label}"])
    ax_city.tick_params(labelsize=textsize)
    # return figure with final styling
    plt.tight_layout()
    fig.savefig(path,dpi=dpi) 
    plt.close(fig)   

def generate_resources(city,gpkg_hexes,df,indicators,comparisons,threshold_scenarios,city_policy,cmap):
    """
    The function prepares a series of image resources required for the global 
    indicator score cards.  These are located in a city specific path, (eg. cities/Melbourne).  This city_path string variable is returned.
    """
    city_path = f'./cities/{city}'
    if not os.path.exists('cities'):
        os.mkdir('cities')
    if not os.path.exists(city_path):
        os.mkdir(os.path.abspath(city_path))
    # read city data
    gdf = gpd.read_file(gpkg_hexes, layer=city.lower().replace(' ','_'))
    gdf['all_cities_walkability'] = gdf['all_cities_walkability'].apply(lambda x:
                                     -6 if x < -6 else (6 if x > 6 else x))
    # Spatial access liveability profile
    profile_title = "Population %\nwith access\nwithin\n500m to..."
    city_stats = {}
    city_stats['access'] = df.loc[city,indicators]
    city_stats_index = city_stats['access'].index.tolist()
    for i,item in enumerate(city_stats['access'].index):
        if str(city_stats['access'][i])=='nan':
            city_stats_index[i] = f"{city_stats['access'].index[i]} (not evaluated)"
    city_stats['access'].index = city_stats_index
    # city stats have NA replaced with zero for li profile to facilitate plotting of comparison
    # it is not that there is no regular pt..
    li_profile(city_stats=city_stats['access'].fillna(0), 
               comparisons = comparisons['access'],
               title=profile_title,
               cmap=cmap, 
               path=f'{city_path}/access_profile.jpg')
    # Spatial distribution maps
    spatial_distribution_figures = [
        {'column' :'all_cities_walkability', 
         'range'    : [-6,6],
         'label'  : 'Neighbourhood walkability relative to 25 global cities\n',
         'tick_labels': ['Low','','','Average','','','High'],
         'outfile': f'{city_path}/all_cities_walkability.jpg'},
        {'column' : 'pct_access_500m_pt_gtfs_freq_20_score', 
         'range'    : [0,100],
         'label'  :('Percentage of population with access to public transport\n'
                   'with service frequency of 20 minutes or less '
                   f'({round(df.loc[city,"Public transport with regular service"],1)}%)'),
         'tick_labels': None,
         'outfile': f'{city_path}/pct_access_500m_pt_gtfs_freq_20_score.jpg'},
        {'column' :'pct_access_500m_public_open_space_large_score', 
         'range'    : [0,100],
         'label'  :('Percentage of population with access to public open\n'
                    'space of area 1.5 hectares or larger '
                    f'({round(df.loc[city,"Large public open space"],1)}%)'),
         'tick_labels': None,
         'outfile': f'{city_path}/pct_access_500m_public_open_space_large_score.jpg'},
    ]
    for f in spatial_distribution_figures:
        spatial_dist_map(gdf,
                     column=f['column'], 
                     range = f['range'],
                     label= f['label'],
                     tick_labels=f['tick_labels'],
                     cmap=cmap,
                     path = f['outfile'])
    # Threshold maps
    for row in threshold_scenarios['data'].index:
        threshold_map(gdf, 
                      column = threshold_scenarios['lookup'][row]['field'], 
                      range = threshold_scenarios['lookup'][row]['range'],
                      scale = threshold_scenarios['lookup'][row]['scale'],
                      comparison = [threshold_scenarios['data'].loc[row].lower, 
                                    threshold_scenarios['data'].loc[row].upper], 
                      label = (f"{threshold_scenarios['lookup'][row]['title']}\n"
             f"({threshold_scenarios['data'].loc[row,city]}% of population within target threshold)"),
             cmap=cmap,
             path = f"{city_path}/{threshold_scenarios['lookup'][row]['field']}.jpg")
             
    # Policy ratings
    policy_rating(
              range = [0,24],
              score = city_policy['Presence_rating'],
              comparison = city_policy['Presence_global'],
              label = '\nPolicies identified',
              cmap=cmap,
              path=f"cities/{city}/policy_presence_rating.jpg")
    policy_rating(
              range = [0,57],
              score = city_policy['Checklist_rating'],
              comparison = city_policy['Checklist_global'],
              label = '',
              cmap=cmap,
              path=f"cities/{city}/policy_checklist_rating.jpg")
    return(city_path)


def pdf_template_setup(csv_template_path):
    """
    Takes a CSV file defining elements for use in fpdf2's FlexTemplate function.
    This is loosely based on the specification at https://pyfpdf.github.io/fpdf2/Templates.html
    However, it has been modified to allow additional definitions which are parsed
    by this function
      - can define the page for which template elements are to be applied
      - colours are specified using standard hexadecimal codes
    Any blank cells are set to represent "None".
    
    The function returns a dictionary of elements, indexed by page number strings.
    """
    # read in elements
    ## NOTE!! It is assumed that CSV has been saved using UTF8 encoding
    ## This allows special characters which may be required.
    elements = pd.read_csv(csv_template_path,encoding = "utf-8")
    document_pages = elements.page.unique()
    elements = elements.to_dict(orient='records')
    elements = [{k:v if not str(v)=='nan' else None for k,v in x.items()} for x in elements]
    
    # Need to convert hexadecimal colours (eg FFFFFF is white) to
    # decimal colours for the fpdf Template class to work
    # We'll establish default hex colours for foreground and background
    planes = {
        'foreground':'000000',
        'background':'FFFFFF'
        }
    
    for i,element in enumerate(elements):
      for plane in planes:
        if elements[i][plane] is not None:
            # this assumes a hexadecimal string without the 0x prefix
            elements[i][plane] = int(elements[i][plane],16)
        else:
            elements[i][plane] = int(planes[plane],16)
    
    pages = {}
    for page in document_pages:
        pages[f'{page}'] = [x for x in elements if x['page']==page]
    return pages

def generate_scorecard(city,pages,title,author,policy_checks):
    policy_indicators = {0:u'✗',0.5:'~',1:u'✓'}
    pdf = FPDF(orientation="portrait", format="A4")
    if debug == True:
        pdf.set_image_filter("DCTDecode")
    pdf.set_title(title)
    pdf.set_author(author)
    pdf.set_auto_page_break(False)
    pdf.add_font('dejavu',style='', fname='fonts/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed.ttf', uni=True)
    pdf.add_font('dejavu',style='B', fname='fonts/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed-Bold.ttf', uni=True)
    pdf.add_font('dejavu',style='I', fname='fonts/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed-Oblique.ttf', uni=True)
    pdf.add_font('dejavu',style='BI', fname='fonts/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed-BoldOblique.ttf', uni=True)
    # Set up Cover page
    pdf.add_page()
    template = FlexTemplate(pdf,elements=pages['1'])
    template["title_city"] = f"{city}"
    if not os.path.exists(f"hero_images/{city}.jpg"):
        template["hero_alt"]="Please provide a high resolution 'hero image' for this city, ideally with dimensions in the ratio of 21:10 (e.g. 2100px by 1000px)"
    else:
        template["hero_image"] = f"hero_images/{city}.jpg"
    template["cover_image"] = "hero_images/cover_background - alt-01.png"
    template.render()
    # Set up next page
    pdf.add_page(orientation="L")
    template = FlexTemplate(pdf,elements=pages['2'])
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf,elements=pages['3'])
    template["access_profile"] = f"cities/{city}/access_profile.jpg"
    template["all_cities_walkability"] = f"cities/{city}/all_cities_walkability.jpg"
    template["local_nh_population_density"] = f"cities/{city}/local_nh_population_density.jpg"
    template["presence_rating"] = f"cities/{city}/policy_presence_rating.jpg"
    template["quality_rating"] = f"cities/{city}/policy_checklist_rating.jpg"
    template["policy2_text1_response"] =policy_indicators[policy_checks['Presence'][0]]
    template["policy2_text2_response"] =policy_indicators[policy_checks['Presence'][1]]
    template["policy2_text3_response"] =policy_indicators[policy_checks['Presence'][2]]
    template["policy2_text4_response"] =policy_indicators[policy_checks['Presence'][3]]
    template["policy2_text5_response"] =policy_indicators[policy_checks['Presence'][4]]
    # Air pollution is a check of two policies met...
    template["policy2_text6_response"] =policy_indicators[(policy_checks['Presence'][5]+policy_checks['Presence'][6])/2].replace('~','½')
    ## Walkable neighbourhood policy checklist
    template["walkability_description"] =f"Walkable neighbourhoods underpin a liveable city, providing opportunities for healthy sustainable lifestyles.  Walkability encompasses accessibility of services and amenities and is influenced by policies determining land use mix and population density, as well as street connectivity. Sufficient density of dwellings and population is critical for walkability, because it determines the viability of local destinations and adequate public transport service.\n\nThe below checklist reports on an analysis of {city} urban policies supporting walkable neighbourhoods, evaluating: policy presence; whether the policy had a specific aim or standard; whether it had a measurable target; and whether it was consistent with evidence on health supportive environments."
    
    for analysis in ['Checklist']:
        for i,policy in enumerate(policy_checks[analysis].index):
            row = i+1
            for j,item in enumerate([x for x in policy_checks[analysis][i][0]]):
                col = j+1
                template[f'policy_{analysis}_text{row}_response{col}'] = item
    
    template.render()
    
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf,elements=pages['4'])
    template["local_nh_intersection_density"] = f"cities/{city}/local_nh_intersection_density.jpg"
    template["pct_access_500m_pt_gtfs_freq_20_score"] = f"cities/{city}/pct_access_500m_pt_gtfs_freq_20_score.jpg"
    template["pct_access_500m_public_open_space_large_score"] = f"cities/{city}/pct_access_500m_public_open_space_large_score.jpg"
    
    for analysis in ['PT','POS']:
        for i,policy in enumerate(policy_checks[analysis].index):
            row = i+1
            for j,item in enumerate([x for x in policy_checks[analysis][i][0]]):
                col = j+1
                template[f'policy_{analysis}_text{row}_response{col}'] = item
    
    template.render()
    
    # Set up last page
    pdf.add_page()
    template = FlexTemplate(pdf,elements=pages['5'])
    template["1024px-RMIT_University_Logo.svg.png"] = f"logos/1024px-RMIT_University_Logo.svg.png"
    template["University_of_Melbourne.png"] = f"logos/University_of_Melbourne.png"
    template["North_Carolina_State_University"] = f"logos/University_of_Melbourne.png"
    template["University_of_Southern_California"] = f"logos/University_of_Melbourne.png"
    template["Australian_Catholic_University"] = f"logos/University_of_Melbourne.png"
    template["University_of_Hong_Kong"] = f"logos/University_of_Melbourne.png"
    template["Auckland_University_of_Technology"] = f"logos/University_of_Melbourne.png"
    template["Northeastern_University"] = f"logos/1024px-RMIT_University_Logo.svg.png"
    template["University_of_California_San_Diego"] = f"logos/University_of_Melbourne.png"
    template["Washington_University_in_St._Louis"] = f"logos/1024px-RMIT_University_Logo.svg.png"
    template["University_of_Washington_Seattle"] = f"logos/University_of_Melbourne.png"
    template["suggested_citation"] = f'Citation: Global Healthy & Sustainable Cities Indicators Collaboration. 2022. Urban Policy and Built Environment Scorecard 2020: Bangkok. https://doi.org/INSERT-DOI-HERE'
    
    template.render()
    
    # Output scorecard pdf
    if debug == True:
        pdf.oversized_images = "DOWNSCALE"
    pdf.output(f"scorecard_{city}.pdf")
    return f"Scorecard generated: scorecard_{city}.pdf"
