# Import required libraries
## Note: requires installation via pip:
##    pip install descartes fpdf2
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
from batlow import batlow_map
import json

# import and set up functions
import scorecard_functions

cmap = batlow_map

generate_resources = False

if __name__ == '__main__':
    # load city parameters
    with open('../../process/configuration/cities.json') as f:
      city_data = json.load(f)
    
    # Identify data sources
    gpkg_cities = os.path.abspath('../../process/data/output/global_indicators_city_2021-06-21.gpkg')
    gpkg_hexes = os.path.abspath('../../process/data/output/global_indicators_hex_250m_2021-06-21.gpkg')
    csv_city_indicators = os.path.abspath("../../process/data/output/global_indicators_city_2021-06-21.csv")
    csv_hex_indicators = os.path.abspath("../../process/data/output/global_indicators_hex_250m_2021-06-21.csv")
    csv_thresholds_data = os.path.abspath("../Global Indicators 2020 - thresholds summary estimates.csv")
    
    # Set up main city indicators
    df = pd.read_csv(csv_city_indicators)
    df.set_index('City',inplace=True)
    vars = {
    'pop_pct_access_500m_fresh_food_market_score': 'Food market',
    'pop_pct_access_500m_convenience_score': 'Convenience',
    'pop_pct_access_500m_public_open_space_any_score': 'Any public open space',
    'pop_pct_access_500m_public_open_space_large_score': 'Large public open space',
    'pop_pct_access_500m_pt_any_score': 'Public transport stop',
    'pop_pct_access_500m_pt_gtfs_freq_20_score': 'Public transport with regular service'
    }   
    df = df.rename(columns=vars)
    indicators = vars.values()
    
    # Set up thresholds
    threshold_lookup = {
            'Mean 1000 m neighbourhood population per km²':{
                'title':'Neighbourhood population density (per km²)',
                'field':'local_nh_population_density',
                'scale':'log'
                 },
            'Mean 1000 m neighbourhood street intersections per km²':{
                'title':'Neighbourhood intersection density (per km²)',
                'field':'local_nh_intersection_density',
                'scale':'log'
                }
            }
    
    # Set up indicator min max summaries
    df_extrema = pd.read_csv(csv_hex_indicators)
    df_extrema.set_index('City',inplace=True) 
    for k in threshold_lookup:
        threshold_lookup[k]['range'] = df_extrema[threshold_lookup[k]['field']].describe()[['min','max']].astype(int).values
    
    threshold_scenarios = scorecard_functions.setup_thresholds(csv_thresholds_data,threshold_lookup)
    
    # Set up between city averages comparisons
    comparisons = {}
    comparisons['access'] = {}
    comparisons['access']['p25'] = df[indicators].quantile(q=.25)
    comparisons['access']['p50'] = df[indicators].median()
    comparisons['access']['p75'] = df[indicators].quantile(q=.75)
    
    # Generate placeholder hero images, if not existing
    #if not os.path.exists('hero_images/{city}.png'):
    
    # Retrieve and parse policy analysis data
    policy_lookup = {
        'worksheet':'data/Policy Figures 1 & 2_23 Dec_numerical.xlsx',
        'analyses': {
            'Presence':{'sheet_name':'Figure 1 - transposed evaluated'},
            'Checklist':{'sheet_name':'Figure 2 - Tuples'}
        },
        'parameters': {'header':[1],'nrows':25,'index_col':2},
        'column_formatting':'Policies of interest'
    }
    
    df_labels = pd.read_excel(policy_lookup['worksheet'],sheet_name = policy_lookup['column_formatting'],index_col=0)
    df_labels = df_labels[~df_labels['Display'].isna()]
    
    df_policy = {}
    
    for policy_analysis in policy_lookup['analyses']:
        df_policy[policy_analysis] = pd.read_excel(
            io = policy_lookup['worksheet'],
            sheet_name = policy_lookup['analyses'][policy_analysis]['sheet_name'],
            header=policy_lookup['parameters']['header'],
            nrows=policy_lookup['parameters']['nrows'],
            index_col=policy_lookup['parameters']['index_col'])
        # only retain relevant columns for this analysis
        df_policy[policy_analysis]=df_policy[policy_analysis][df_labels[df_labels['Display']==policy_analysis].index]
    
    # parse checklist   
    df_policy['Checklist'] = df_policy['Checklist'].apply(lambda x: x.str.split(':'),axis=1)
    
    # Loop over cities
    cities = ['Bangkok','Melbourne','Mexico City']
    for city in cities:
        print(city)
        year = 2020
        # Generate resources
        if generate_resources:
            scorecard_functions.generate_resources(city,gpkg_hexes,df,indicators,comparisons,threshold_scenarios,cmap)
        
        city_policy={}
        for policy_analysis in policy_lookup['analyses']:
             city_policy[policy_analysis] = df_policy[policy_analysis].loc[city]
        
        #Generate PDF reports for cities
        pages = scorecard_functions.pdf_template_setup('scorecard_template_elements.csv')
        
        # mock up for policy evaluation function
        def policy_checklist():
            policy_checks = [0,0,1]
            return policy_checks
        
        policy_checks = policy_checklist()
        
        # instantiate template
        
        scorecard_functions.generate_scorecard(
            city,
            pages,
            title = f"{city} Global Liveability Indicators Scorecard - {year}",
            author = 'Global Healthy Liveable City Indicators Collaboaration Study',
            policy_checks = city_policy
            )
