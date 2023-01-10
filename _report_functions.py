"""
Report functions.

Define functions used for formatting and saving indicator reports.
"""
import json
import os
import time
from textwrap import wrap

import fiona
import geopandas as gpd
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from babel.numbers import format_decimal as fnum
from babel.units import format_unit
from fpdf import FPDF, FlexTemplate
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from batlow import batlow_map as cmap


def get_and_setup_language_cities(config):
    """Setup and return languages for given configuration."""
    if config.auto_language:
        languages = pd.read_excel(config.configuration, sheet_name="languages")
        languages = languages[languages["name"] == config.city].dropna(
            axis=1, how="all"
        )
        languages = list(languages.columns[2:])
    else:
        languages = [config.language]
    return languages


def generate_report_for_language(
    config, language, indicators, regions, policies
):
    """
    Generate report for a processed city in a given language.
    """
    # try:
    city = config.city
    font = get_and_setup_font(language, config)
    # set up policies
    city_policy = policy_data_setup(policies, regions[city]["policy_review"])
    # get city and grid summary data
    gpkg = regions[city]["gpkg"]
    layers = fiona.listlayers(gpkg)
    gdf_city = gpd.read_file(
        gpkg,
        layer=[
            l
            for l in layers
            if l.startswith(
                regions[city]["city_summary"].strip(time.strftime("%Y-%m-%d"))
            )
        ][0],
    )
    gdf_grid = gpd.read_file(
        gpkg,
        layer=[
            l
            for l in layers
            if l.startswith(
                regions[city]["grid_summary"].strip(time.strftime("%Y-%m-%d"))
            )
        ][0],
    )
    ### a proposed empirical walkability-related target (not used):
    ### Percentage of population who live in a neighbourhood with walkable access
    ### to a food market, large public open space, and a public transport stop within 500 metres
    ## indicators['report']['thresholds']['walkability_target'] = round(
    ##    100*gdf_grid.query(
    ##        'pct_access_500m_fresh_food_market_score == 100 and '
    ##        'pct_access_500m_public_open_space_large_score == 100 and '
    ##        'pct_access_500m_pt_any_score == 100'
    ##        )['pop_est']\
    ##            .sum()/gdf_grid['pop_est'].sum(),
    ##    1)
    #
    # The below currently relates walkability to the GHSCIC 25 city median (as per study)
    # returns tuple of GeoDataFrame and summary percentage for percentage of pop > median pct
    gdf_grid = evaluate_comparative_walkability(
        gdf_grid, indicators["report"]["walkability"]["ghscic_reference"]
    )
    indicators["report"]["walkability"][
        "walkability_above_median_pct"
    ] = evaluate_threshold_pct(
        gdf_grid,
        "all_cities_walkability",
        ">",
        indicators["report"]["walkability"]["ghscic_walkability_reference"],
    )
    for i in indicators["report"]["thresholds"]:
        indicators["report"]["thresholds"][i]["pct"] = evaluate_threshold_pct(
            gdf_grid,
            indicators["report"]["thresholds"][i]["field"],
            indicators["report"]["thresholds"][i]["relationship"],
            indicators["report"]["thresholds"][i]["criteria"],
        )
    # set up phrases
    phrases = prepare_phrases(config, city, language, regions)
    # Generate resources
    if config.generate_resources:
        capture_return = generate_resources(
            config,
            gdf_city,
            gdf_grid,
            phrases,
            indicators,
            regions,
            city_policy,
            language,
            cmap,
        )
    # instantiate template
    for template in config.templates:
        print(f" [{template}]")
        capture_return = generate_scorecard(
            config, phrases, indicators, city_policy, language, template, font,
        )
    print(capture_return)
    # except Exception as e:
    # print(f"\t- Report generation failed with error: {e}")


def get_and_setup_font(language, config):
    """Setup and return font for given language configuration."""
    fonts = pd.read_excel(config.configuration, sheet_name="fonts")
    if language.replace(" (Auto-translation)", "") in fonts.Language.unique():
        fonts = fonts.loc[
            fonts["Language"] == language.replace(" (Auto-translation)", "")
        ].fillna("")
    else:
        fonts = fonts.loc[fonts["Language"] == "default"].fillna("")
    main_font = fonts.File.values[0].strip()
    fm.fontManager.addfont(main_font)
    prop = fm.FontProperties(fname=main_font)
    fm.findfont(prop=prop, directory=main_font, rebuild_if_missing=True)
    plt.rcParams["font.family"] = prop.get_name()
    font = fonts.Font.values[0]
    return font


def policy_data_setup(policies, policy_review):
    """
    Returns a dictionary of policy data
    """
    review = pd.read_excel(policy_review, index_col=0)
    df_policy = {}
    # Presence score
    df_policy["Presence_rating"] = review.loc["Score"]["Policy identified"]
    # Quality score
    df_policy["Checklist_rating"] = review.loc["Score"]["Quality"]
    # Presence
    df_policy["Presence"] = review.loc[
        [p["Policy"] for p in policies if p["Display"] == "Presence"]
    ].apply(lambda x: x["Weight"] * x["Policy identified"], axis=1)
    # GDP
    df_policy["Presence_gdp"] = pd.DataFrame(
        [
            {
                c: p[c]
                for c in p
                if c
                in ["Label", "gdp_comparison_middle", "gdp_comparison_upper"]
            }
            for p in policies
            if p["Display"] == "Presence"
        ]
    )
    df_policy["Presence_gdp"].columns = ["Policy", "middle", "upper"]
    df_policy["Presence_gdp"].set_index("Policy", inplace=True)
    # Urban Checklist
    df_policy["Checklist"] = review.loc[
        [p["Policy"] for p in policies if p["Display"] == "Checklist"]
    ]["Checklist"]
    # Public open space checklist
    df_policy["POS"] = review.loc[
        [p["Policy"] for p in policies if p["Display"] == "POS"]
    ]["Checklist"]
    # Public transport checklist
    df_policy["PT"] = review.loc[
        [p["Policy"] for p in policies if p["Display"] == "PT"]
    ]["Checklist"]
    return df_policy


def evaluate_comparative_walkability(gdf_grid, reference):
    # Evaluate walkability relative to 25-city study reference
    for x in reference:
        gdf_grid[f"z_{x}"] = (gdf_grid[x] - reference[x]["mean"]) / reference[
            x
        ]["sd"]
    gdf_grid["all_cities_walkability"] = sum(
        [gdf_grid[f"z_{x}"] for x in reference]
    )
    return gdf_grid


def evaluate_threshold_pct(
    gdf_grid, indicator, relationship, reference, population="pop_est"
):
    """
    Evaluate whether a pandas series meets a threshold criteria (eg. '<' or '>'
    """
    percentage = round(
        100
        * gdf_grid.query(f"{indicator} {relationship} {reference}")[
            population
        ].sum()
        / gdf_grid[population].sum(),
        1,
    )
    return percentage


def generate_resources(
    config,
    gdf_city,
    gdf_grid,
    phrases,
    indicators,
    regions,
    city_policy,
    language,
    cmap,
):
    """
    The function prepares a series of image resources required for the global
    indicator score cards.  These are located in a city specific path, (eg. cities/Melbourne).  This city_path string variable is returned.
    """
    figure_path = f"{config.city_path}/figures"
    locale = phrases["locale"]
    city_stats = compile_city_stats(gdf_city, indicators, phrases)
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    # Spatial access liveability profile
    li_profile(
        city_stats=city_stats,
        title=phrases["Population % with access within 500m to..."],
        cmap=cmap,
        phrases=phrases,
        path=f"{figure_path}/access_profile_{language}.jpg",
    )
    ## constrain extreme outlying walkability for representation
    gdf_grid["all_cities_walkability"] = gdf_grid[
        "all_cities_walkability"
    ].apply(lambda x: -6 if x < -6 else (6 if x > 6 else x))
    # Spatial distribution maps
    spatial_maps = compile_spatial_map_info(
        indicators["report"]["spatial_distribution_figures"],
        gdf_city,
        phrases,
        locale,
        language=language,
    )
    for f in spatial_maps:
        spatial_dist_map(
            gdf_grid,
            column=f,
            range=spatial_maps[f]["range"],
            label=spatial_maps[f]["label"],
            tick_labels=spatial_maps[f]["tick_labels"],
            cmap=cmap,
            path=f'{figure_path}/{spatial_maps[f]["outfile"]}',
            phrases=phrases,
            locale=locale,
        )
    # Threshold maps
    for scenario in indicators["report"]["thresholds"]:
        threshold_map(
            gdf_grid,
            column=indicators["report"]["thresholds"][scenario]["field"],
            scale=indicators["report"]["thresholds"][scenario]["scale"],
            comparison=indicators["report"]["thresholds"][scenario][
                "criteria"
            ],
            label=(
                f"{phrases[indicators['report']['thresholds'][scenario]['title']]} ({phrases['density_units']})"
            ),
            cmap=cmap,
            path=f"{figure_path}/{indicators['report']['thresholds'][scenario]['field']}_{language}.jpg",
            phrases=phrases,
            locale=locale,
        )
    # Policy ratings
    policy_rating(
        range=[0, 24],
        score=city_policy["Presence_rating"],
        comparison=indicators["report"]["policy"]["comparisons"]["presence"],
        label="",
        comparison_label=phrases["25 city comparison"],
        cmap=cmap,
        locale=locale,
        path=f"{figure_path}/policy_presence_rating_{language}.jpg",
    )
    policy_rating(
        range=[0, 57],
        score=city_policy["Checklist_rating"],
        comparison=indicators["report"]["policy"]["comparisons"]["quality"],
        label="",
        comparison_label=phrases["25 city comparison"],
        cmap=cmap,
        locale=locale,
        path=f"{figure_path}/policy_checklist_rating_{language}.jpg",
    )
    return figure_path


def fpdf2_mm_scale(mm):
    # returns a width double that of the conversion of mm to inches
    # because that seems to work about right, based on trial and error
    return 2 * mm / 25.4


def _pct(value, locale, length="short"):
    return format_unit(value, "percent", locale=locale, length=length)


def compile_city_stats(gdf_city, indicators, phrases):
    """Compile a set of city statistics with comparisons, given a processed geodataframe of city summary statistics and a dictionary of indicators including reference percentiles."""
    city_stats = {}
    city_stats["access"] = gdf_city[
        indicators["report"]["accessibility"].keys()
    ].transpose()[0]
    city_stats["access"].index = [
        indicators["report"]["accessibility"][x]["title"]
        if city_stats["access"][x] is not None
        else f"{indicators['report']['accessibility'][x]['title']} (not evaluated)"
        for x in city_stats["access"].index
    ]
    city_stats["access"] = city_stats["access"].fillna(
        0
    )  # for display purposes
    city_stats["comparisons"] = {
        indicators["report"]["accessibility"][x]["title"]: indicators[
            "report"
        ]["accessibility"][x]["ghscic_reference"]
        for x in indicators["report"]["accessibility"]
    }
    city_stats["percentiles"] = {}
    for percentile in ["p25", "p50", "p75"]:
        city_stats["percentiles"][percentile] = [
            city_stats["comparisons"][x][percentile]
            for x in city_stats["comparisons"].keys()
        ]
    city_stats["access"].index = [
        phrases[x] for x in city_stats["access"].index
    ]
    return city_stats


def compile_spatial_map_info(
    spatial_distribution_figures, gdf_city, phrases, locale, language
):
    """Compile required information to produce spatial distribution figures, given the dictionary:  indicators['report']['spatial_distribution_figures']"""
    # effectively deep copy the supplied dictionary so its not mutable
    spatial_maps = json.loads(json.dumps(spatial_distribution_figures))
    for i in spatial_maps:
        for text in ["label", "outfile"]:
            spatial_maps[i][text] = spatial_maps[i][text].format(**locals())
        if spatial_maps[i]["tick_labels"] is not None:
            spatial_maps[i]["tick_labels"] = [
                x.format(**{"phrases": phrases})
                for x in spatial_maps[i]["tick_labels"]
            ]
        if i.startswith("pct_"):
            city_summary_percent = _pct(
                fnum(gdf_city[f"pop_{i}"].fillna(0)[0], "0.0", locale), locale
            )
            spatial_maps[i][
                "label"
            ] = f'{spatial_maps[i]["label"]} ({city_summary_percent})'
    if gdf_city["pop_pct_access_500m_pt_gtfs_freq_20_score"][0] is None:
        spatial_maps["pct_access_500m_pt_any_score"] = spatial_maps.pop(
            "pct_access_500m_pt_gtfs_freq_20_score"
        )
        spatial_maps["pct_access_500m_pt_any_score"]["label"] = (
            f'{phrases["Percentage of population with access to public transport"]}\n'
            f'({_pct(fnum(gdf_city["pop_pct_access_500m_pt_any_score"][0],"0.0",locale),locale)})'
        )
    return spatial_maps


def add_scalebar(
    ax,
    length,
    multiplier,
    units,
    fontproperties,
    loc="upper left",
    pad=0,
    color="black",
    frameon=False,
    size_vertical=2,
    locale="en",
):
    """
    Adds a scalebar to matplotlib map.
    Requires import of: from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    As a rule of thumb, a scalebar of 1/3 of feature size seems appropriate.
    For example, to achieve this, calculate the variable 'length' as

        gdf_width = gdf.geometry.total_bounds[2] - gdf.geometry.total_bounds[0]
        scalebar_length = int(gdf_width / (3000))
    """
    scalebar = AnchoredSizeBar(
        ax.transData,
        length * multiplier,
        format_unit(length, units, locale=locale, length="short"),
        loc=loc,
        pad=pad,
        color=color,
        frameon=frameon,
        size_vertical=size_vertical,
        fontproperties=fontproperties,
    )
    ax.add_artist(scalebar)


def add_localised_north_arrow(
    ax,
    text="N",
    xy=(1, 0.96),
    textsize=14,
    arrowprops=dict(facecolor="black", width=4, headwidth=8),
):
    """
    Add a minimal north arrow with custom text label (eg 'N' or other language equivalent) above it
    to a matplotlib map.  Default placement is in upper right corner of map.
    """
    arrow = ax.annotate(
        "",
        xy=(1, 0.96),
        xycoords=ax.transAxes,
        xytext=(0, -0.5),
        textcoords="offset pixels",
        va="center",
        ha="center",
        arrowprops=arrowprops,
    )
    ax.annotate(
        text,
        xy=(0.5, 1.5),
        xycoords=arrow,
        va="center",
        ha="center",
        fontsize=textsize,
    )


## radar chart
def li_profile(
    city_stats,
    title,
    cmap,
    path,
    phrases,
    width=fpdf2_mm_scale(80),
    height=fpdf2_mm_scale(80),
    dpi=300,
):
    """
    Generates a radar chart for city liveability profiles
    Expanding on https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib
    -- A python code blog post by Yan Holtz, in turn expanding on work of Tomás Capretto and Tobias Stadler.

    Arguments:
        city_stats  A pandas series of indicators for a particular city, including comparisons
        cmap A colour map
    """
    figsize = (width, height)
    # Values for the x axis
    ANGLES = np.linspace(
        0.15, 2 * np.pi - 0.05, len(city_stats["access"]), endpoint=False
    )
    VALUES = city_stats["access"].values
    COMPARISON = city_stats["percentiles"]["p50"]
    INDICATORS = city_stats["access"].index
    # Colours
    GREY12 = "#1f1f1f"
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    COLORS = cmap(list(norm(VALUES)))
    # Initialize layout in polar coordinates
    textsize = 11
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
    ax.vlines(
        ANGLES,
        city_stats["percentiles"]["p25"],
        city_stats["percentiles"]["p75"],
        color=GREY12,
        zorder=11,
    )
    # Add dots to represent the mean gain
    comparison_text = "\n".join(
        wrap(phrases["25 city comparison"], 17, break_long_words=False)
    )
    ax.scatter(
        ANGLES,
        COMPARISON,
        s=60,
        color=GREY12,
        zorder=11,
        label=comparison_text,
    )
    # Add labels for the indicators
    try:
        LABELS = [
            "\n".join(wrap(r, 12, break_long_words=False)) for r in INDICATORS
        ]
    except Exception:
        LABELS = INDICATORS
    # Set the labels
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(LABELS, size=textsize)
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
    PAD = 0
    for num in [0, 50, 100]:
        ax.text(
            -0.2 * np.pi / 2,
            num + PAD,
            f"{num}%",
            ha="center",
            va="center",
            backgroundcolor="white",
            size=textsize,
        )
    # Add text to explain the meaning of the height of the bar and the
    # height of the dot
    ax.text(
        ANGLES[0],
        -50,
        "\n".join(wrap(title, 13, break_long_words=False)),
        rotation=0,
        ha="center",
        va="center",
        size=textsize,
        zorder=12,
    )
    angle = np.deg2rad(130)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.58 + np.cos(angle) / 2, 0.46 + np.sin(angle) / 2),
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


## Spatial distribution mapping
def spatial_dist_map(
    gdf,
    column,
    range,
    label,
    tick_labels,
    cmap,
    path,
    width=fpdf2_mm_scale(88),
    height=fpdf2_mm_scale(80),
    dpi=300,
    phrases={"north arrow": "N", "km": "km"},
    locale="en",
):
    """
    Spatial distribution maps using geopandas geodataframe
    """
    figsize = (width, height)
    textsize = 14
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    divider = make_axes_locatable(ax)  # Define 'divider' for the axes
    # Legend axes will be located at the 'bottom' of figure, with width '5%' of ax and
    # a padding between them equal to '0.1' inches
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        vmin=range[0],
        vmax=range[1],
        legend_kwds={
            "label": "\n".join(wrap(label, 60, break_long_words=False))
            if label.find("\n") < 0
            else label,
            "orientation": "horizontal",
        },
        cax=cax,
        cmap=cmap,
    )
    # scalebar
    add_scalebar(
        ax,
        length=int(
            (gdf.geometry.total_bounds[2] - gdf.geometry.total_bounds[0])
            / (3000)
        ),
        multiplier=1000,
        units="kilometer",
        locale=locale,
        fontproperties=fm.FontProperties(size=textsize),
    )
    # north arrow
    add_localised_north_arrow(ax, text=phrases["north arrow"])
    # axis formatting
    cax.tick_params(labelsize=textsize)
    cax.xaxis.label.set_size(textsize)
    if tick_labels is not None:
        # cax.set_xticks(cax.get_xticks().tolist())
        # cax.set_xticklabels(tick_labels)
        cax.xaxis.set_major_locator(ticker.MaxNLocator(len(tick_labels)))
        ticks_loc = cax.get_xticks().tolist()
        cax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        cax.set_xticklabels(tick_labels)
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def threshold_map(
    gdf,
    column,
    comparison,
    scale,
    label,
    cmap,
    path,
    width=fpdf2_mm_scale(88),
    height=fpdf2_mm_scale(80),
    dpi=300,
    phrases={"north arrow": "N", "km": "km"},
    locale="en",
):
    """Create threshold indicator map."""
    figsize = (width, height)
    textsize = 14
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    divider = make_axes_locatable(ax)  # Define 'divider' for the axes
    # Legend axes will be located at the 'bottom' of figure, with width '5%' of ax and
    # a padding between them equal to '0.1' inches
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        legend_kwds={
            "label": "\n".join(wrap(label, 60, break_long_words=False))
            if label.find("\n") < 0
            else label,
            "orientation": "horizontal",
        },
        cax=cax,
        cmap=cmap,
    )
    # scalebar
    add_scalebar(
        ax,
        length=int(
            (gdf.geometry.total_bounds[2] - gdf.geometry.total_bounds[0])
            / (3000)
        ),
        multiplier=1000,
        units="kilometer",
        locale=locale,
        fontproperties=fm.FontProperties(size=textsize),
    )
    # north arrow
    add_localised_north_arrow(ax, text=phrases["north arrow"])
    # axis formatting
    cax.xaxis.set_major_formatter(ticker.EngFormatter())
    cax.tick_params(labelsize=textsize)
    cax.xaxis.label.set_size(textsize)
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def policy_rating(
    range,
    score,
    cmap,
    comparison=None,
    width=fpdf2_mm_scale(70),
    height=fpdf2_mm_scale(15),
    label="Policies identified",
    comparison_label="25 city median",
    locale="en",
    path="policy_rating_test.jpg",
    dpi=300,
):
    """
    Plot a score (policy rating) and optional comparison (e.g. 25 cities median score) on
    a colour bar.  Applied in this context for policy presence and policy quality scores.
    """
    textsize = 14
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(bottom=0)
    cmap = cmap
    norm = mpl.colors.Normalize(vmin=range[0], vmax=range[1])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        # shrink=0.9, pad=0, aspect=90
    )
    # Format Global ticks
    if comparison is None:
        ax.xaxis.set_ticks([])
    else:
        ax.xaxis.set_major_locator(ticker.FixedLocator([comparison]))
        # ax.set_xticklabels([comparison_label])
        ax.set_xticklabels([""])
        ax.tick_params(labelsize=textsize)
        ax.plot(
            comparison,
            0,
            marker="v",
            color="black",
            markersize=9,
            zorder=10,
            clip_on=False,
        )
        if comparison < 7:
            for t in ax.get_yticklabels():
                t.set_horizontalalignment("left")
        if comparison > 18:
            for t in ax.get_yticklabels():
                t.set_horizontalalignment("right")
    # Format City ticks
    ax_city = ax.twiny()
    ax_city.set_xlim(range)
    ax_city.xaxis.set_major_locator(ticker.FixedLocator([score]))
    ax_city.plot(
        score,
        1,
        marker="^",
        color="black",
        markersize=9,
        zorder=10,
        clip_on=False,
    )
    sep = ""
    # if comparison is not None and label=='':
    ax_city.set_xticklabels(
        [f"{sep}{str(score).rstrip('0').rstrip('.')}/{range[1]}{label}"]
    )
    ax_city.tick_params(labelsize=textsize)
    # return figure with final styling
    xlabel = f"{comparison_label} ({fnum(comparison,'0.0',locale)})"
    ax.set_xlabel(
        xlabel, labelpad=0.5, fontsize=textsize,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def pdf_template_setup(
    config, template="template_web", font=None, language="English",
):
    """
    Takes a template xlsx sheet defining elements for use in fpdf2's FlexTemplate function.
    This is loosely based on the specification at https://pyfpdf.github.io/fpdf2/Templates.html
    However, it has been modified to allow additional definitions which are parsed
    by this function
      - can define the page for which template elements are to be applied
      - colours are specified using standard hexadecimal codes
    Any blank cells are set to represent "None".
    The function returns a dictionary of elements, indexed by page number strings.
    """
    # read in elements
    elements = pd.read_excel(config.configuration, sheet_name=template)
    document_pages = elements.page.unique()
    # Conditional formatting to help avoid inappropriate line breaks and gaps in Tamil and Thai
    if language in ["Tamil", "Thai"]:
        elements["align"] = elements["align"].replace("J", "L")
        elements.loc[
            (elements["type"] == "T") & (elements["size"] < 12), "size"
        ] = (
            elements.loc[
                (elements["type"] == "T") & (elements["size"] < 12), "size"
            ]
            - 1
        )
    if font is not None:
        elements.loc[elements.font == "custom", "font"] = font
    elements = elements.to_dict(orient="records")
    elements = [
        {k: v if not str(v) == "nan" else None for k, v in x.items()}
        for x in elements
    ]
    # Need to convert hexadecimal colours (eg FFFFFF is white) to
    # decimal colours for the fpdf Template class to work
    # We'll establish default hex colours for foreground and background
    planes = {"foreground": "000000", "background": "FFFFFF"}
    for i, element in enumerate(elements):
        for plane in planes:
            if elements[i][plane] is not None:
                # this assumes a hexadecimal string without the 0x prefix
                elements[i][plane] = int(elements[i][plane], 16)
            else:
                elements[i][plane] = int(planes[plane], 16)
    pages = {}
    for page in document_pages:
        pages[f"{page}"] = [x for x in elements if x["page"] == page]
    return pages


def format_pages(pages, phrases):
    """Format pages with phrases."""
    for page in pages:
        for i, item in enumerate(pages[page]):
            if item["name"] in phrases:
                try:
                    pages[page][i]["text"] = phrases[item["name"]].format(
                        city=phrases["city_name"],
                        country=phrases["country_name"],
                        study_doi=phrases["study_doi"],
                        citation_series=phrases["citation_series"],
                        citation_doi=phrases["citation_doi"],
                        citation_population=phrases["citation_population"],
                        citation_boundaries=phrases["citation_boundaries"],
                        citation_features=phrases["citation_features"],
                        citation_colour=phrases["citation_colour"],
                    )
                except Exception:
                    pages[f"{page}"][i]["text"] = phrases[item["name"]]
    return pages


def prepare_phrases(config, city, language, regions):
    """Prepare dictionary for specific language translation given English phrase."""
    languages = pd.read_excel(config.configuration, sheet_name="languages")
    phrases = json.loads(languages.set_index("name").to_json())[language]
    city_details = pd.read_excel(
        config.configuration, sheet_name="city_details", index_col="City"
    )
    country_code = regions[city]["region"]
    # set default English country code
    if language == "English" and country_code not in ["AU", "GB", "US"]:
        country_code = "AU"
    phrases["locale"] = f'{phrases["language_code"]}_{country_code}'
    # extract English language variables
    phrases["metadata_author"] = languages.loc[
        languages["name"] == "title_author", "English"
    ].values[0]
    phrases["metadata_title1"] = languages.loc[
        languages["name"] == "title_series_line1", "English"
    ].values[0]
    phrases["metadata_title2"] = languages.loc[
        languages["name"] == "title_series_line2", "English"
    ].values[0]
    phrases["country"] = languages.loc[
        languages["name"] == f"{city} - Country", "English"
    ].values[0]
    # restrict to specific language
    languages = languages.loc[
        languages["role"] == "template", ["name", language]
    ]
    phrases["vernacular"] = languages.loc[
        languages["name"] == "language", language
    ].values[0]
    phrases["city_name"] = languages.loc[
        languages["name"] == city, language
    ].values[0]
    phrases["country_name"] = languages.loc[
        languages["name"] == f"{city} - Country", language
    ].values[0]
    phrases["city"] = city
    phrases["study_doi"] = f'https://doi.org/{city_details["DOI"]["Study"]}'
    phrases["city_doi"] = f'https://doi.org/{city_details["DOI"][city]}'
    phrases["study_executive_names"] = city_details["Names"]["Study"]
    phrases["local_collaborators_names"] = city_details["Names"][city]
    phrases["credit_image1"] = city_details["credit_image1"][city]
    phrases["credit_image2"] = city_details["credit_image2"][city]
    # incoporating study citations
    citation_json = json.loads(city_details["exceptions_json"]["Study"])
    # handle city-specific exceptions
    city_exceptions = json.loads(city_details["exceptions_json"][city])
    if language in city_exceptions:
        city_exceptions = json.loads(
            city_exceptions[language].replace("'", '"')
        )
        for e in city_exceptions:
            phrases[e] = city_exceptions[e].replace("|", "\n")
    for citation in citation_json:
        if citation != "citation_doi" or "citation_doi" not in phrases:
            phrases[citation] = (
                citation_json[citation].replace("|", "\n").format(**phrases)
            )
    phrases["citation_doi"] = phrases["citation_doi"].format(**phrases)
    return phrases


def wrap_sentences(words, limit=50, delimiter=""):
    """Wrap sentences if exceeding limit."""
    sentences = []
    sentence = ""
    gap = len(delimiter)
    for i, word in enumerate(words):
        if i == 0:
            sentence = word
            continue
        # combine word to sentence if under limit
        if len(sentence) + gap + len(word) <= limit:
            sentence = sentence + delimiter + word
        else:
            sentences.append(sentence)
            sentence = word
            # append the final word if not yet appended
            if i == len(words) - 1:
                sentences.append(sentence)
        # finally, append sentence of all words if still below limit
        if (i == len(words) - 1) and (sentences == []):
            sentences.append(sentence)
    return sentences


def prepare_pdf_fonts(pdf, config, language):
    """Prepare PDF fonts."""
    fonts = pd.read_excel(config.configuration, sheet_name="fonts")
    fonts = (
        fonts.loc[
            fonts["Language"].isin(
                ["default", language.replace(" (Auto-translation)", "")]
            )
        ]
        .fillna("")
        .drop_duplicates()
    )
    for s in ["", "B", "I", "BI"]:
        for langue in ["default", language]:
            if (
                langue.replace(" (Auto-translation)", "")
                in fonts.Language.unique()
            ):
                f = fonts.loc[
                    (
                        fonts["Language"]
                        == langue.replace(" (Auto-translation)", "")
                    )
                    & (fonts["Style"] == s)
                ]
                if f"{f.Font.values[0]}{s}" not in pdf.fonts.keys():
                    pdf.add_font(
                        f.Font.values[0], style=s, fname=f.File.values[0]
                    )


def save_pdf_layout(pdf, folder, template, filename):
    """
    Save a PDF report in template subfolder in specified location.
    """
    if not os.path.exists(folder):
        os.mkdir(folder)
    template_folder = f"{folder}/_{template} reports"
    if not os.path.exists(template_folder):
        os.mkdir(template_folder)
    pdf.output(f"{template_folder}/{filename}")
    return f"Scorecard generated ({template_folder}):\n{filename}\n"


def generate_scorecard(
    config,
    phrases,
    indicators,
    city_policy,
    language="English",
    template="web",
    font=None,
):
    """
    Format a PDF using the pyfpdf FPDF2 library, and drawing on definitions from a UTF-8 CSV file.

    Included in this function is the marking of a policy 'scorecard', with ticks, crosses, etc.
    """
    locale = phrases["locale"]
    # Set up PDF document template pages
    pages = pdf_template_setup(config, "template_web", font, language,)
    pages = format_pages(pages, phrases)
    # initialise PDF
    pdf = FPDF(orientation="portrait", format="A4", unit="mm")
    # set up fonts
    prepare_pdf_fonts(pdf, config, language)
    pdf.set_author(phrases["metadata_author"])
    pdf.set_title(f"{phrases['metadata_title1']} {phrases['metadata_title2']}")
    pdf.set_auto_page_break(False)
    if template == "web":
        pdf = pdf_for_web(
            pdf,
            pages,
            config,
            language,
            locale,
            phrases,
            indicators,
            city_policy,
        )
    elif template == "print":
        pdf = pdf_for_print(
            pdf,
            pages,
            config,
            language,
            locale,
            phrases,
            indicators,
            city_policy,
        )
    # Output report pdf
    filename = f"{phrases['city_name']} - {phrases['title_series_line1'].replace(':','')} - GHSCIC 2022 - {phrases['vernacular']}.pdf"
    if phrases["_export"] == 1:
        capture_result = save_pdf_layout(
            pdf, folder=config.city_path, template=template, filename=filename,
        )
        return capture_result
    else:
        return "Skipped."


def pdf_for_web(
    pdf, pages, config, language, locale, phrases, indicators, city_policy,
):
    city = config.city
    city_path = config.city_path
    figure_path = f"{city_path}/figures"
    # Set up Cover page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["1"])
    if os.path.exists(f"{city_path}/hero_images/{city}-1.jpg"):
        template["hero_image"] = f"{city_path}/hero_images/{city}-1.jpg"
        template["hero_alt"] = ""
        template["credit_image1"] = phrases["credit_image1"]
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["2"])
    template["citations"] = phrases["citations"]
    template["study_executive_names"] = phrases["study_executive_names"]
    template["local_collaborators"] = template["local_collaborators"].format(
        title_city=phrases["title_city"]
    )
    template["local_collaborators_names"] = phrases[
        "local_collaborators_names"
    ]
    if phrases["translation_names"] is None:
        template["translation"] = ""
        template["translation_names"] = ""
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["3"])
    template[
        "introduction"
    ] = f"{phrases['series_intro']}\n\n{phrases['series_interpretation']}".format(
        **phrases
    )
    ## Access profile plot
    template["access_profile"] = f"{figure_path}/access_profile_{language}.jpg"
    ## Walkability plot
    template[
        "all_cities_walkability"
    ] = f"{figure_path}/all_cities_walkability_{language}.jpg"
    template["walkability_above_median_pct"] = phrases[
        "walkability_above_median_pct"
    ].format(
        _pct(
            fnum(
                indicators["report"]["walkability"][
                    "walkability_above_median_pct"
                ],
                "0.0",
                locale,
            ),
            locale,
        )
    )
    ## Policy ratings
    template[
        "presence_rating"
    ] = f"{figure_path}/policy_presence_rating_{language}.jpg"
    template[
        "quality_rating"
    ] = f"{figure_path}/policy_checklist_rating_{language}.jpg"
    template["city_header"] = phrases["city_name"]
    ## City planning requirement presence (round 0.5 up to 1)
    policy_indicators = {0: "✗", 0.5: "~", 1: "✓"}
    for x in range(1, 7):
        # check presence
        template[f"policy_urban_text{x}_response"] = policy_indicators[
            np.ceil(city_policy["Presence"][x - 1])
        ]
        # format percentage units according to locale
        for gdp in ["middle", "upper"]:
            template[f"policy_urban_text{x}_{gdp}"] = _pct(
                float(city_policy["Presence_gdp"].iloc[x - 1][gdp]),
                locale,
                length="short",
            )
    ## Walkable neighbourhood policy checklist
    for i, policy in enumerate(city_policy["Checklist"].index):
        row = i + 1
        for j, item in enumerate([x for x in city_policy["Checklist"][i][0]]):
            col = j + 1
            template[f"policy_{'Checklist'}_text{row}_response{col}"] = item
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["4"])
    ## Density plots
    template[
        "local_nh_population_density"
    ] = f"{figure_path}/local_nh_population_density_{language}.jpg"
    template[
        "local_nh_intersection_density"
    ] = f"{figure_path}/local_nh_intersection_density_{language}.jpg"
    ## Density threshold captions
    for scenario in indicators["report"]["thresholds"]:
        template[scenario] = phrases[f"optimal_range - {scenario}"].format(
            _pct(
                fnum(
                    indicators["report"]["thresholds"][scenario]["pct"],
                    "0.0",
                    locale,
                ),
                locale,
            ),
            fnum(
                indicators["report"]["thresholds"][scenario]["criteria"],
                "#,000",
                locale,
            ),
            phrases["density_units"],
        )
    if os.path.exists(f"{city_path}/hero_images/{city}-2.jpg"):
        template["hero_image_2"] = f"{city_path}/hero_images/{city}-2.jpg"
        template["hero_alt_2"] = ""
        template["credit_image2"] = phrases["credit_image2"]
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["5"])
    template[
        "pct_access_500m_pt.jpg"
    ] = f"{figure_path}/pct_access_500m_pt_{language}.jpg"
    template[
        "pct_access_500m_public_open_space_large_score"
    ] = f"{figure_path}/pct_access_500m_public_open_space_large_score_{language}.jpg"
    template["city_text"] = phrases[f"{city} - Summary"]
    ## Checklist ratings for PT and POS
    for analysis in ["PT", "POS"]:
        for i, policy in enumerate(city_policy[analysis].index):
            row = i + 1
            for j, item in enumerate([x for x in city_policy[analysis][i][0]]):
                col = j + 1
                template[f"policy_{analysis}_text{row}_response{col}"] = item
    template.render()
    # Set up last page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["6"])
    template.render()
    return pdf


def pdf_for_print(
    pdf, pages, config, language, locale, phrases, indicators, city_policy,
):
    city = config.city
    city_path = config.city_path
    figure_path = f"{city_path}/figures"
    # Set up Cover page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["1"])
    if os.path.exists(f"{city_path}/hero_images/{city}-1.jpg"):
        template["hero_image"] = f"{city_path}/hero_images/{city}-1.jpg"
        template["hero_alt"] = ""
        template["credit_image1"] = phrases["credit_image1"]
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["2"])
    template["citations"] = phrases["citations"]
    template["study_executive_names"] = phrases["study_executive_names"]
    template["local_collaborators"] = template["local_collaborators"].format(
        title_city=phrases["title_city"]
    )
    template["local_collaborators_names"] = phrases[
        "local_collaborators_names"
    ]
    if phrases["translation_names"] is None:
        template["translation"] = ""
        template["translation_names"] = ""
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["3"])
    template[
        "introduction"
    ] = f"{phrases['series_intro']}\n\n{phrases['series_interpretation']}".format(
        **phrases
    )
    ## Policy ratings
    template[
        "presence_rating"
    ] = f"{figure_path}/policy_presence_rating_{language}.jpg"
    template[
        "quality_rating"
    ] = f"{figure_path}/policy_checklist_rating_{language}.jpg"
    ## Access profile plot
    template["access_profile"] = f"{figure_path}/access_profile_{language}.jpg"
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["4"])
    ## City planning requirement presence (round 0.5 up to 1)
    template["city_header"] = phrases["city_name"]
    policy_indicators = {0: "✗", 0.5: "~", 1: "✓"}
    for x in range(1, 7):
        # check presence
        template[f"policy_urban_text{x}_response"] = policy_indicators[
            np.ceil(city_policy["Presence"][x - 1])
        ]
        # format percentage units according to locale
        for gdp in ["middle", "upper"]:
            template[f"policy_urban_text{x}_{gdp}"] = _pct(
                float(city_policy["Presence_gdp"].iloc[x - 1][gdp]),
                locale,
                length="short",
            )
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["5"])
    ## Walkability plot
    template[
        "all_cities_walkability"
    ] = f"{figure_path}/all_cities_walkability_{language}.jpg"
    template["walkability_above_median_pct"] = phrases[
        "walkability_above_median_pct"
    ].format(
        _pct(
            fnum(
                indicators["report"]["walkability"][
                    "walkability_above_median_pct"
                ],
                "0.0",
                locale,
            ),
            locale,
        )
    )
    ## Walkable neighbourhood policy checklist
    for i, policy in enumerate(city_policy["Checklist"].index):
        row = i + 1
        for j, item in enumerate([x for x in city_policy["Checklist"][i][0]]):
            col = j + 1
            template[f"policy_{'Checklist'}_text{row}_response{col}"] = item
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["6"])
    ## Density plots
    template[
        "local_nh_population_density"
    ] = f"{figure_path}/local_nh_population_density_{language}.jpg"
    template[
        "local_nh_intersection_density"
    ] = f"{figure_path}/local_nh_intersection_density_{language}.jpg"
    ## Density threshold captions
    for scenario in indicators["report"]["thresholds"]:
        template[scenario] = phrases[f"optimal_range - {scenario}"].format(
            _pct(
                fnum(
                    indicators["report"]["thresholds"][scenario]["pct"],
                    "0.0",
                    locale,
                ),
                locale,
            ),
            fnum(
                indicators["report"]["thresholds"][scenario]["criteria"],
                "#,000",
                locale,
            ),
            phrases["density_units"],
        )
    if os.path.exists(f"{city_path}/hero_images/{city}-2.jpg"):
        template["hero_image_2"] = f"{city_path}/hero_images/{city}-2.jpg"
        template["hero_alt_2"] = ""
        template["credit_image2"] = phrases["credit_image2"]
    template.render()
    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["7"])
    template[
        "pct_access_500m_pt.jpg"
    ] = f"{figure_path}/pct_access_500m_pt_{language}.jpg"
    template[
        "pct_access_500m_public_open_space_large_score"
    ] = f"{figure_path}/pct_access_500m_public_open_space_large_score_{language}.jpg"
    template["city_text"] = phrases[f"{city} - Summary"]
    ## Checklist ratings for PT and POS
    for analysis in ["PT", "POS"]:
        for i, policy in enumerate(city_policy[analysis].index):
            row = i + 1
            for j, item in enumerate([x for x in city_policy[analysis][i][0]]):
                col = j + 1
                template[f"policy_{analysis}_text{row}_response{col}"] = item
    template.render()
    # Set up last page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["8"])
    template["licence_image"] = "logos/by-nc.jpg"
    template.render()
    return pdf
