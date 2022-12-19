"""
Report functions.

Define functions used for formatting and saving indicator reports.
"""
import json
import os
from textwrap import wrap

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


def get_and_setup_language_cities(config):
    """Setup and return languages for given configuration."""
    if config.auto_language:
        languages = pd.read_excel(config.configuration, sheet_name="languages")
        languages = languages.query(f"name in {cities}").dropna(
            axis=1, how="all"
        )
        languages = languages[languages.columns[1:]].set_index("name")
        # replace all city name variants with English equivalent for grouping purposes
        for city in languages.index:
            languages.loc[city][languages.loc[city].notnull()] = city
        languages = (
            languages[languages.columns]
            .transpose()
            .stack()
            .groupby(level=0)
            .apply(list)
        )
    else:
        cities = [x.strip() for x in config.cities.split(",")]
        languages = pd.Series([cities], index=[config.language])
    return languages


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


def generate_report_for_language(config, language, cities, indicators):
    """Generate report for cities corresponding to language configuration."""
    print(f"\n{language} language reports:")

    # set up fonts
    font = get_and_setup_font(language, config)

    # Set up main city indicators
    data_setup = indicators["report"]["data"]
    df = pd.read_csv(_data_setup.csv_city_indicators)
    df.set_index("City", inplace=True)
    df = df.rename(columns=indicators["report"]["accessibility"])

    # Set up indicator min max summaries
    df_extrema = pd.read_csv(_data_setup.csv_hex_indicators)
    df_extrema.set_index("City", inplace=True)
    for k in indicators["report"]["thresholds"]:
        indicators["report"]["thresholds"][k]["range"] = (
            df_extrema[indicators["report"]["thresholds"][k]["field"]]
            .describe()[["min", "max"]]
            .astype(int)
            .values
        )

    threshold_scenarios = setup_thresholds(
        _data_setup.csv_thresholds_data, indicators["report"]["thresholds"]
    )

    # Set up between city averages comparisons
    comparisons = {}
    comparisons["access"] = {}
    comparisons["access"]["p25"] = df[
        indicators["report"]["accessibility"].values()
    ].quantile(q=0.25)
    comparisons["access"]["p50"] = df[
        indicators["report"]["accessibility"].values()
    ].median()
    comparisons["access"]["p75"] = df[
        indicators["report"]["accessibility"].values()
    ].quantile(q=0.75)

    # Generate placeholder hero images, if not existing
    # if not os.path.exists('hero_images/{city}.png'):

    df_policy = policy_data_setup(policy_lookup=_data_setup.policy_lookup)

    walkability_stats = pd.read_csv(
        _data_setup.csv_walkability_data, index_col="City"
    )

    # Loop over cities
    successful = 0
    for city in cities:
        print(f"\n- {city}"),
        try:
            year = 2020
            city_policy = {}
            for policy_analysis in _data_setup.policy_lookup["analyses"]:
                city_policy[policy_analysis] = df_policy[policy_analysis].loc[
                    city
                ]
                if policy_analysis in ["Presence", "Checklist"]:
                    city_policy[f"{policy_analysis}_rating"] = df_policy[
                        f"{policy_analysis}_rating"
                    ].loc[city]
                    city_policy[f"{policy_analysis}_global"] = df_policy[
                        f"{policy_analysis}_rating"
                    ].describe()

            city_policy["Presence_gdp"] = df_policy["Presence_gdp"]
            threshold_scenarios["walkability"] = walkability_stats.loc[
                city, "pct_walkability_above_median"
            ]
            # set up phrases
            phrases = prepare_phrases(config, city, language)

            # Generate resources
            if config.generate_resources:
                capture_return = generate_resources(
                    city,
                    phrases,
                    _data_setup.gpkg_hexes,
                    df,
                    indicators["report"]["accessibility"].values(),
                    comparisons,
                    threshold_scenarios,
                    city_policy,
                    language,
                    cmap,
                )

            # instantiate template
            for template in templates:
                print(f" [{template}]")
                capture_return = generate_scorecard(
                    city,
                    phrases,
                    threshold_scenarios=threshold_scenarios,
                    city_policy=city_policy,
                    config=config,
                    template=template,
                    language=language,
                    font=font,
                )

            successful += 1
        except Exception as e:
            print(f"\t- Report generation failed with error: {e}")

        print(f"\n {successful}/{len(cities)} cities processed successfully!")


def fpdf2_mm_scale(mm):
    # returns a width double that of the conversion of mm to inches
    # because that seems to work about right, based on trial and error
    return 2 * mm / 25.4


def _pct(value, locale, length="short"):
    return format_unit(value, "percent", locale=locale, length=length)


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
    comparisons,
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
        city_stats  A pandas series of indicators for a particular city
        comparisons A dictionary of pandas series IQR point summaries (p25, p50 and p75)
        cmap A colour map
    """
    figsize = (width, height)
    # Values for the x axis
    ANGLES = np.linspace(
        0.15, 2 * np.pi - 0.05, len(city_stats), endpoint=False
    )
    VALUES = city_stats.values
    COMPARISON = comparisons["p50"].values
    INDICATORS = city_stats.index
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
        ANGLES, comparisons["p25"], comparisons["p75"], color=GREY12, zorder=11
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


def setup_thresholds(csv_thresholds_data, threshold_lookup, scenario="B"):
    """
    A help script for identifying lower and upper bound thresholds associated with specific policy scenarios
    """
    thresholds = pd.read_csv(csv_thresholds_data)
    cities = thresholds.columns[1:]
    # Threshold analysis
    threshold_index_cols = ["scenario", "description", "location"]
    thresholds["scenario"] = thresholds.iloc[:, 0].apply(
        lambda x: x.split(" - ")[0].split("_")[-1]
    )
    thresholds["description"] = thresholds.iloc[:, 0].apply(
        lambda x: x.split(" - ")[1]
    )
    thresholds["location"] = thresholds.iloc[:, 0].apply(
        lambda x: x.split(" - ")[2]
    )
    thresholds = thresholds[threshold_index_cols + list(cities)]
    # Extract threshold scenarios from variables
    threshold_scenarios = thresholds.loc[
        ~(thresholds.location.str.startswith("below"))
        & (thresholds.scenario == scenario)
    ]
    threshold_lower_bound = threshold_scenarios.loc[
        threshold_scenarios.location.str.startswith("within"),
        ["description", "location"],
    ]
    threshold_lower_bound["location"] = threshold_lower_bound.location.apply(
        lambda x: int(x[0:-1].split("(")[1].split(", ")[0])
    )
    threshold_lower_bound = threshold_lower_bound.set_index("description")
    threshold_scenarios = threshold_scenarios.groupby(["description"]).sum(
        numeric_only=True
    )
    threshold_scenarios = {
        "data": threshold_scenarios,
        "lookup": threshold_lookup,
        "lower_bound": threshold_lower_bound,
    }
    return threshold_scenarios


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
        ax.xaxis.set_major_locator(ticker.FixedLocator([comparison["50%"]]))
        # ax.set_xticklabels([comparison_label])
        ax.set_xticklabels([""])
        ax.tick_params(labelsize=textsize)
        ax.plot(
            comparison["50%"],
            0,
            marker="v",
            color="black",
            markersize=9,
            zorder=10,
            clip_on=False,
        )
        if comparison["50%"] < 7:
            for t in ax.get_yticklabels():
                t.set_horizontalalignment("left")
        if comparison["50%"] > 18:
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
    xlabel = f"{comparison_label} ({fnum(comparison['50%'],'0.0',locale)})"
    ax.set_xlabel(
        xlabel, labelpad=0.5, fontsize=textsize,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def generate_resources(
    city,
    phrases,
    gpkg_hexes,
    df,
    indicators,
    comparisons,
    threshold_scenarios,
    city_policy,
    language,
    cmap,
):
    """
    The function prepares a series of image resources required for the global
    indicator score cards.  These are located in a city specific path, (eg. cities/Melbourne).  This city_path string variable is returned.
    """
    locale = phrases["locale"]
    # read city data
    gdf = gpd.read_file(gpkg_hexes, layer=city.lower().replace(" ", "_"))
    gdf["all_cities_walkability"] = gdf["all_cities_walkability"].apply(
        lambda x: -6 if x < -6 else (6 if x > 6 else x)
    )
    # create output directory for plots
    city_path = f"./cities/{city}"
    if not os.path.exists("cities"):
        os.mkdir("cities")
    if not os.path.exists(city_path):
        os.mkdir(city_path)
    # Spatial access liveability profile
    city_stats = {}
    city_stats["access"] = df.loc[city, indicators]
    city_stats_index = city_stats["access"].index.tolist()
    for i, item in enumerate(city_stats["access"].index):
        if str(city_stats["access"][i]) == "nan":
            city_stats_index[
                i
            ] = f"{city_stats['access'].index[i]} (not evaluated)"
    city_stats["access"].index = city_stats_index
    city_stats["access"].index = [
        phrases[x] for x in city_stats["access"].index
    ]
    # city stats have NA replaced with zero for li profile to facilitate plotting of comparison
    # it is not that there is no regular pt..
    profile_title = phrases["Population % with access within 500m to..."]
    li_profile(
        city_stats=city_stats["access"].fillna(0),
        comparisons=comparisons["access"],
        title=profile_title,
        cmap=cmap,
        phrases=phrases,
        path=f"{city_path}/access_profile_{language}.jpg",
    )
    # Spatial distribution maps
    spatial_distribution_figures = [
        {
            "column": "all_cities_walkability",
            "range": [-6, 6],
            "label": f'{phrases["Neighbourhood walkability relative to 25 global cities"]}\n',
            "tick_labels": [
                phrases["Low"],
                "",
                "",
                phrases["Average"],
                "",
                "",
                phrases["High"],
            ],
            "outfile": f"{city_path}/all_cities_walkability_{language}.jpg",
        },
        {
            "column": "pct_access_500m_pt_gtfs_freq_20_score",
            "range": [0, 100],
            "label": (
                f"{phrases[('Percentage of population with access to public transport with service frequency of 20 minutes or less')]} "
                f'({fnum(df.loc[city,"Public transport with regular service"],"0.0",locale)}%)'
            ),
            "tick_labels": None,
            "outfile": f"{city_path}/pct_access_500m_pt_{language}.jpg",
        },
        {
            "column": "pct_access_500m_public_open_space_large_score",
            "range": [0, 100],
            "label": (
                f"{phrases[('Percentage of population with access to public open space of area 1.5 hectares or larger')]} "
                f'({_pct(fnum(df.loc[city,"Large public open space"],"0.0",locale),locale)})'
            ),
            "tick_labels": None,
            "outfile": f"{city_path}/pct_access_500m_public_open_space_large_score_{language}.jpg",
        },
    ]
    if (
        "pct_access_500m_pt_gtfs_freq_20_score"
        not in gdf.describe().transpose().index
    ):
        spatial_distribution_figures[1][
            "column"
        ] = "pct_access_500m_pt_any_score"
        spatial_distribution_figures[1]["label"] = (
            f"{phrases['Percentage of population with access to public transport']}\n"
            f'({_pct(fnum(df.loc[city,"Public transport stop"],"0.0",locale),locale)})'
        )
    for f in spatial_distribution_figures:
        spatial_dist_map(
            gdf,
            column=f["column"],
            range=f["range"],
            label=f["label"],
            tick_labels=f["tick_labels"],
            cmap=cmap,
            path=f["outfile"],
            phrases=phrases,
            locale=locale,
        )
    # Threshold maps
    for row in threshold_scenarios["data"].index:
        threshold_map(
            gdf,
            column=threshold_scenarios["lookup"][row]["field"],
            scale=threshold_scenarios["lookup"][row]["scale"],
            comparison=threshold_scenarios["lower_bound"].loc[row].location,
            label=(
                f"{phrases[threshold_scenarios['lookup'][row]['title']]} ({phrases['density_units']})"
            ),
            cmap=cmap,
            path=f"{city_path}/{threshold_scenarios['lookup'][row]['field']}_{language}.jpg",
            phrases=phrases,
            locale=locale,
        )
    # Policy ratings
    policy_rating(
        range=[0, 24],
        score=city_policy["Presence_rating"],
        comparison=city_policy["Presence_global"],
        label="",
        comparison_label=phrases["25 city comparison"],
        cmap=cmap,
        locale=locale,
        path=f"{city_path}/policy_presence_rating_{language}.jpg",
    )
    policy_rating(
        range=[0, 57],
        score=city_policy["Checklist_rating"],
        comparison=city_policy["Checklist_global"],
        label="",
        comparison_label=phrases["25 city comparison"],
        cmap=cmap,
        locale=locale,
        path=f"{city_path}/policy_checklist_rating_{language}.jpg",
    )
    return city_path


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


def prepare_phrases(config, city, language):
    """Prepare dictionary for specific language translation given English phrase."""
    languages = pd.read_excel(config.configuration, sheet_name="languages")
    phrases = json.loads(languages.set_index("name").to_json())[language]
    city_details = pd.read_excel(
        config.configuration, sheet_name="city_details"
    )
    city_details = json.loads(city_details.set_index("City").to_json())
    country_code = city_details["Country Code"][city]
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


def save_pdf_layout(
    pdf, folder, template, language, city, by_city, by_language, filename
):
    """
    Save a PDF report in city, language, and template specific folder locations.
    """
    if not os.path.exists(folder):
        os.mkdir(folder)

    template_folder = f"{folder}/{template} reports"
    if not os.path.exists(template_folder):
        os.mkdir(template_folder)

    paths = []
    if by_city:
        if not os.path.exists(f"{template_folder}/by_city"):
            os.mkdir(f"{template_folder}/by_city")
        paths.append(f"{template_folder}/by_city/{city}")

    if by_language:
        paths.append(f"{template_folder}/{language}")

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

        pdf.output(f"{path}/{filename}")

    return f"Scorecard generated ({paths}): {filename}"


def generate_scorecard(
    city,
    phrases,
    city_policy,
    threshold_scenarios,
    config,
    language="English",
    template="template_web",
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

    if template.startswith("template_web"):
        pdf = pdf_for_web(
            pdf,
            pages,
            city,
            language,
            locale,
            phrases,
            threshold_scenarios,
            city_policy,
        )
    elif template.startswith("template_print"):
        pdf = pdf_for_print(
            pdf,
            pages,
            city,
            language,
            locale,
            phrases,
            threshold_scenarios,
            city_policy,
        )

    # Output report pdf
    filename = f"{phrases['city_name']} - {phrases['title_series_line1'].replace(':','')} - GHSCIC 2022 - {phrases['vernacular']}.pdf"
    if phrases["_export"] == 1:
        capture_result = save_pdf_layout(
            pdf,
            folder="scorecards",
            template=f'{template.replace("template", "")}',
            language=language,
            city=city,
            by_city=config.by_city,
            by_language=config.by_language,
            filename=filename,
        )
        return capture_result
    else:
        return "Skipped."


def pdf_for_web(
    pdf,
    pages,
    city,
    language,
    locale,
    phrases,
    threshold_scenarios,
    city_policy,
):
    # Set up Cover page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["1"])
    if os.path.exists(f"hero_images/{city}-1.jpg"):
        template["hero_image"] = f"hero_images/{city}-1.jpg"
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
    template["access_profile"] = f"cities/{city}/access_profile_{language}.jpg"
    ## Walkability plot
    template[
        "all_cities_walkability"
    ] = f"cities/{city}/all_cities_walkability_{language}.jpg"
    template["walkability_above_median_pct"] = phrases[
        "walkability_above_median_pct"
    ].format(
        _pct(fnum(threshold_scenarios["walkability"], "0.0", locale), locale)
    )
    ## Policy ratings
    template[
        "presence_rating"
    ] = f"cities/{city}/policy_presence_rating_{language}.jpg"
    template[
        "quality_rating"
    ] = f"cities/{city}/policy_checklist_rating_{language}.jpg"
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
    ] = f"cities/{city}/local_nh_population_density_{language}.jpg"

    template[
        "local_nh_intersection_density"
    ] = f"cities/{city}/local_nh_intersection_density_{language}.jpg"

    ## Density threshold captions
    for row in threshold_scenarios["data"].index:
        template[row] = phrases[f"optimal_range - {row}"].format(
            _pct(
                fnum(
                    threshold_scenarios["data"].loc[row, city], "0.0", locale
                ),
                locale,
            ),
            fnum(
                threshold_scenarios["lower_bound"].loc[row].location,
                "#,000",
                locale,
            ),
            phrases["density_units"],
        )

    if os.path.exists(f"hero_images/{city}-2.jpg"):
        template["hero_image_2"] = f"hero_images/{city}-2.jpg"
        template["hero_alt_2"] = ""
        template["credit_image2"] = phrases["credit_image2"]

    template.render()

    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["5"])
    template[
        "pct_access_500m_pt.jpg"
    ] = f"cities/{city}/pct_access_500m_pt_{language}.jpg"
    template[
        "pct_access_500m_public_open_space_large_score"
    ] = f"cities/{city}/pct_access_500m_public_open_space_large_score_{language}.jpg"
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
    pdf,
    pages,
    city,
    language,
    locale,
    phrases,
    threshold_scenarios,
    city_policy,
):
    # Set up Cover page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["1"])
    if os.path.exists(f"hero_images/{city}-1.jpg"):
        template["hero_image"] = f"hero_images/{city}-1.jpg"
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
    ] = f"cities/{city}/policy_presence_rating_{language}.jpg"
    template[
        "quality_rating"
    ] = f"cities/{city}/policy_checklist_rating_{language}.jpg"

    ## Access profile plot
    template["access_profile"] = f"cities/{city}/access_profile_{language}.jpg"

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
    ] = f"cities/{city}/all_cities_walkability_{language}.jpg"
    template["walkability_above_median_pct"] = phrases[
        "walkability_above_median_pct"
    ].format(
        _pct(fnum(threshold_scenarios["walkability"], "0.0", locale), locale)
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
    ] = f"cities/{city}/local_nh_population_density_{language}.jpg"

    template[
        "local_nh_intersection_density"
    ] = f"cities/{city}/local_nh_intersection_density_{language}.jpg"

    ## Density threshold captions
    for row in threshold_scenarios["data"].index:
        template[row] = phrases[f"optimal_range - {row}"].format(
            _pct(
                fnum(
                    threshold_scenarios["data"].loc[row, city], "0.0", locale
                ),
                locale,
            ),
            fnum(
                threshold_scenarios["lower_bound"].loc[row].location,
                "#,000",
                locale,
            ),
            phrases["density_units"],
        )

    if os.path.exists(f"hero_images/{city}-2.jpg"):
        template["hero_image_2"] = f"hero_images/{city}-2.jpg"
        template["hero_alt_2"] = ""
        template["credit_image2"] = phrases["credit_image2"]

    template.render()

    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["7"])
    template[
        "pct_access_500m_pt.jpg"
    ] = f"cities/{city}/pct_access_500m_pt_{language}.jpg"
    template[
        "pct_access_500m_public_open_space_large_score"
    ] = f"cities/{city}/pct_access_500m_public_open_space_large_score_{language}.jpg"
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


def policy_data_setup(policy_lookup):
    """
    Returns a pretty complicated dictionary of policy data,
    formatted according to the policy lookup configuration json.
    Should be simplified, really.
    """
    df_labels = pd.read_excel(
        policy_lookup["worksheet"],
        sheet_name=policy_lookup["column_formatting"],
        index_col=0,
    )
    df_labels = df_labels[~df_labels["Display"].isna()].sort_values(
        by=["Display", "Order"]
    )

    df_policy = {}

    for policy_analysis in policy_lookup["analyses"]:
        df_policy[policy_analysis] = pd.read_excel(
            io=policy_lookup["worksheet"],
            sheet_name=policy_lookup["analyses"][policy_analysis][
                "sheet_name"
            ],
            header=policy_lookup["parameters"]["header"],
            nrows=policy_lookup["parameters"]["nrows"],
            index_col=policy_lookup["parameters"]["index_col"],
        )
        if policy_analysis == "Presence":
            # get percentage of policies meeting requirements stratified by income GDP groups
            df_policy[f"{policy_analysis}_gdp"] = round(
                (
                    100
                    * df_policy["Presence"]
                    .loc[:, df_policy["Presence"].columns[:-1]]
                    .replace(0.5, 1)
                    .groupby(df_policy["Presence"]["GDP"] == "High-income")[
                        df_policy["Presence"].columns[2:-1]
                    ]
                    .mean()
                    .transpose()
                ),
                0,
            )
            df_policy[f"{policy_analysis}_gdp"].columns = [
                "middle",
                "upper",
            ]
            # restrict to policies of interest
            df_policy[f"{policy_analysis}_gdp"] = df_policy[
                f"{policy_analysis}_gdp"
            ].loc[
                [
                    x
                    for x in df_labels.loc[
                        df_labels["Display"] == "Presence"
                    ].index
                    if x in df_policy[f"{policy_analysis}_gdp"].index
                ]
            ]
            # format with short labels
            df_policy[f"{policy_analysis}_gdp"].index = df_labels.loc[
                df_policy[f"{policy_analysis}_gdp"].index, "Label"
            ].values
        if policy_analysis in ["Presence", "Checklist"]:
            # store overall rating for this analysis
            df_policy[f"{policy_analysis}_rating"] = df_policy[
                policy_analysis
            ].loc[:, policy_lookup["analyses"][policy_analysis]["column"]]
        # only retain relevant columns for this analysis
        df_policy[policy_analysis] = df_policy[policy_analysis][
            df_labels[df_labels["Display"] == policy_analysis].index
        ]
        if policy_analysis != "Presence":
            # parse checklist
            df_policy[policy_analysis] = df_policy[policy_analysis].apply(
                lambda x: x.str.split(":"), axis=1
            )
    return df_policy
