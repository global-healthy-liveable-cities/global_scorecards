# scorecard functions
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
from fpdf import FPDF, FlexTemplate
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def fpdf2_mm_scale(mm):
    # returns a width double that of the conversion of mm to inches
    # because that seems to work about right, based on trial and error
    return 2 * mm / 25.4


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
        0.05, 2 * np.pi - 0.05, len(city_stats), endpoint=False
    )
    VALUES = city_stats.values
    COMPARISON = comparisons["p50"].values
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
            "\n".join(wrap(r, 10, break_long_words=False)) for r in INDICATORS
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
    PAD = -2.3
    for num in [0, 50, 100]:
        ax.text(
            -0.2 * np.pi / 2, num + PAD, f"{num}", ha="center", size=textsize
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
            "label": "\n".join(wrap(label, 60, break_long_words=False)),
            "orientation": "horizontal",
        },
        cax=cax,
        cmap=cmap,
    )
    gdf_width = gdf.geometry.total_bounds[2] - gdf.geometry.total_bounds[0]
    # cities vary in scale, but in general
    # a scalebar no bigger than 1/3 of the city is ideally
    # so is rounding to nearest thousand (ie. units of km)
    # and I think the following expression meets these goals
    scalebar_length = int(gdf_width / (3000))
    scalebar = AnchoredSizeBar(
        ax.transData,
        scalebar_length * 1000,
        f"{scalebar_length} km",
        "upper left",
        pad=0,
        color="black",
        frameon=False,
        size_vertical=2,
        fontproperties=fm.FontProperties(size=textsize),
    )
    ax.add_artist(scalebar)
    x, y, arrow_length = 1, 1, 0.06
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(facecolor="black", width=4, headwidth=8),
        ha="center",
        va="center",
        fontsize=textsize,
        xycoords=ax.transAxes,
    )
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
):
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
            "label": "\n".join(wrap(label, 60, break_long_words=False)),
            "orientation": "horizontal",
        },
        cax=cax,
        cmap=cmap,
    )
    gdf_width = gdf.geometry.total_bounds[2] - gdf.geometry.total_bounds[0]
    # cities vary in scale, but in general
    # a scalebar no bigger than 1/3 of the city is ideally
    # so is rounding to nearest thousand (ie. units of km)
    # and I think the following expression meets these goals
    scalebar_length = int(gdf_width / (3000))
    scalebar = AnchoredSizeBar(
        ax.transData,
        scalebar_length * 1000,
        f"{scalebar_length} km",
        "upper left",
        pad=0,
        color="black",
        frameon=False,
        size_vertical=2,
        fontproperties=fm.FontProperties(size=textsize),
    )
    ax.add_artist(scalebar)
    x, y, arrow_length = 1, 1, 0.06
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(facecolor="black", width=4, headwidth=8),
        ha="center",
        va="center",
        fontsize=textsize,
        xycoords=ax.transAxes,
    )
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
    threshold_scenarios = threshold_scenarios.groupby(["description"]).sum()
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
    ax.set_xlabel(
        f"{comparison_label} ({comparison['50%']})",
        labelpad=0.5,
        fontsize=textsize,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def generate_resources(
    city,
    gpkg_hexes,
    df,
    indicators,
    comparisons,
    threshold_scenarios,
    city_policy,
    xlsx_scorecard_template,
    language,
    cmap,
):
    """
    The function prepares a series of image resources required for the global
    indicator score cards.  These are located in a city specific path, (eg. cities/Melbourne).  This city_path string variable is returned.
    """
    city_path = f"./cities/{city}"
    if not os.path.exists("cities"):
        os.mkdir("cities")
    if not os.path.exists(city_path):
        os.mkdir(city_path)
    # read city data
    gdf = gpd.read_file(gpkg_hexes, layer=city.lower().replace(" ", "_"))
    gdf["all_cities_walkability"] = gdf["all_cities_walkability"].apply(
        lambda x: -6 if x < -6 else (6 if x > 6 else x)
    )
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
    languages = pd.read_excel(xlsx_scorecard_template, sheet_name="languages")
    languages = languages.loc[languages["role"] == "plot", ["name", language]]
    phrases = json.loads(languages.set_index("name").to_json())[language]
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
                f'({df.loc[city,"Public transport with regular service"]:.1f}%)'
            ),
            "tick_labels": None,
            "outfile": f"{city_path}/pct_access_500m_pt_{language}.jpg",
        },
        {
            "column": "pct_access_500m_public_open_space_large_score",
            "range": [0, 100],
            "label": (
                f"{phrases[('Percentage of population with access to public open space of area 1.5 hectares or larger')]} "
                f'({df.loc[city,"Large public open space"]:.1f}%)'
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
            f'({df.loc[city,"Public transport stop"]:.1f}%)'
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
        )

    # Policy ratings
    policy_rating(
        range=[0, 24],
        score=city_policy["Presence_rating"],
        comparison=city_policy["Presence_global"],
        label="",
        comparison_label=phrases["25 city comparison"],
        cmap=cmap,
        path=f"{city_path}/policy_presence_rating_{language}.jpg",
    )
    policy_rating(
        range=[0, 57],
        score=city_policy["Checklist_rating"],
        comparison=city_policy["Checklist_global"],
        label="",
        comparison_label=phrases["25 city comparison"],
        cmap=cmap,
        path=f"{city_path}/policy_checklist_rating_{language}.jpg",
    )
    return city_path


def pdf_template_setup(
    template, template_sheet="scorecard_template_elements", font=None
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
    elements = pd.read_excel(template, sheet_name=template_sheet)
    document_pages = elements.page.unique()
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


def generate_scorecard(
    city,
    year,
    pages,
    city_policy,
    threshold_scenarios,
    xlsx_scorecard_template,
    language="English",
    template_sheet="scorecard_template_elements",
):
    """
    Format a PDF using the pyfpdf FPDF2 library, and drawing on definitions from a UTF-8 CSV file.

    Included in this function is the marking of a policy 'scorecard', with ticks, crosses, etc.
    """
    # replace placeholder language with specific language, if specified
    if not os.path.exists("scorecards"):
        os.mkdir("scorecards")

    languages = pd.read_excel(xlsx_scorecard_template, sheet_name="languages")
    credits = pd.read_excel(xlsx_scorecard_template, sheet_name="credits")
    credits = json.loads(credits.set_index("City").to_json())
    phrases = json.loads(languages.set_index("name").to_json())[language]
    # extract English language variables
    metadata_author = languages.loc[
        languages["name"] == "title_author", "English"
    ].values[0]
    metadata_title1 = languages.loc[
        languages["name"] == "title_series_line1", "English"
    ].values[0]
    metadata_title2 = languages.loc[
        languages["name"] == "title_series_line2", "English"
    ].values[0]
    country = languages.loc[
        languages["name"] == f"{city} - Country", "English"
    ].values[0]
    # restrict to specific language
    languages = languages.loc[
        languages["role"] == "template", ["name", language]
    ]
    vernacular = languages.loc[
        languages["name"] == "language", language
    ].values[0]
    city_name = languages.loc[languages["name"] == city, language].values[0]
    country_name = languages.loc[
        languages["name"] == f"{city} - Country", language
    ].values[0]
    for p in pages:
        for i, item in enumerate(pages[p]):
            if item["name"] in languages.name.values:
                pages[p][i]["text"] = str(
                    languages.loc[
                        languages["name"] == item["name"], language
                    ].values[0]
                ).format(
                    city=city_name,
                    country=country_name,
                    year=year,
                    citation_series=phrases["citation_series"],
                    citation_doi=phrases["citation_doi"].format(
                        city=city, country=country, language=vernacular,
                    ),
                    citation_population=phrases["citation_population"],
                    citation_boundaries=phrases["citation_boundaries"],
                    citation_features=phrases["citation_features"],
                    citation_colour=phrases["citation_colour"],
                )

    scorecard_path = f"scorecards/{language}"
    if not os.path.exists(scorecard_path):
        os.mkdir(scorecard_path)

    policy_indicators = {0: "✗", 0.5: "~", 1: "✓"}
    pdf = FPDF(orientation="portrait", format="A4")

    fonts = pd.read_excel(xlsx_scorecard_template, sheet_name="fonts")
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
                        f.Font.values[0],
                        style=s,
                        fname=f.File.values[0],
                        uni=True,
                    )

    pdf.set_author(metadata_author)
    pdf.set_title(f"{metadata_title1} {metadata_title2}")
    pdf.set_auto_page_break(False)

    # Set up Cover page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["1"])
    template["title_city"] = f"{city_name}, {country_name}"
    # template["title_year"] = f"{year}"
    if os.path.exists(f"hero_images/{city}-1.jpg"):
        template["hero_image"] = f"hero_images/{city}-1.jpg"
        template["hero_alt"] = ""
        template["hero_credit"] = credits["Image 1 Credit"][city]

    template["cover_image"] = "hero_images/cover_background - alt-01.png"
    template.render()

    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["2"])
    template["title_city"] = f"{city_name}, {country_name}"
    ## Access profile plot
    template["access_profile"] = f"cities/{city}/access_profile_{language}.jpg"
    ## Walkability plot
    template[
        "all_cities_walkability"
    ] = f"cities/{city}/all_cities_walkability_{language}.jpg"
    template["walkability_above_median_pct"] = phrases[
        "walkability_above_median_pct"
    ].format(threshold_scenarios["walkability"])
    ## Policy ratings
    template[
        "presence_rating"
    ] = f"cities/{city}/policy_presence_rating_{language}.jpg"
    template[
        "quality_rating"
    ] = f"cities/{city}/policy_checklist_rating_{language}.jpg"
    template["city_header"] = city_name
    ## City planning requirement presence (round 0.5 up to 1)
    template["policy_urban_text1_response"] = policy_indicators[
        np.ceil(city_policy["Presence"][0])
    ]
    template["policy_urban_text2_response"] = policy_indicators[
        np.ceil(city_policy["Presence"][1])
    ]
    template["policy_urban_text3_response"] = policy_indicators[
        np.ceil(city_policy["Presence"][2])
    ]
    template["policy_urban_text4_response"] = policy_indicators[
        np.ceil(city_policy["Presence"][3])
    ]
    template["policy_urban_text5_response"] = policy_indicators[
        np.ceil(city_policy["Presence"][4])
    ]
    template["policy_urban_text6_response"] = policy_indicators[
        np.ceil(city_policy["Presence"][5])
    ]

    ## Walkable neighbourhood policy checklist
    for analysis in ["Checklist"]:
        for i, policy in enumerate(city_policy[analysis].index):
            row = i + 1
            for j, item in enumerate([x for x in city_policy[analysis][i][0]]):
                col = j + 1
                template[f"policy_{analysis}_text{row}_response{col}"] = item

    template.render()

    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["3"])
    template["title_city"] = f"{city_name}, {country_name}"

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
            threshold_scenarios["data"].loc[row, city],
            threshold_scenarios["lower_bound"].loc[row].location,
            phrases["density_units"],
        )

    if os.path.exists(f"hero_images/{city}-2.jpg"):
        template["hero_image_2"] = f"hero_images/{city}-2.jpg"
        template["hero_alt_2"] = ""
        template["hero_credit"] = credits["Image 2 Credit"][city]

    template.render()

    # Set up next page
    pdf.add_page()
    template = FlexTemplate(pdf, elements=pages["4"])
    template["title_city"] = f"{city_name}, {country_name}"
    template[
        "pct_access_500m_pt.jpg"
    ] = f"cities/{city}/pct_access_500m_pt_{language}.jpg"
    template[
        "pct_access_500m_public_open_space_large_score"
    ] = f"cities/{city}/pct_access_500m_public_open_space_large_score_{language}.jpg"
    template[
        "city_text"
    ] = f"{languages.loc[languages['name']==f'{city} - Summary',language].values[0]}"

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
    template = FlexTemplate(pdf, elements=pages["5"])
    template["citations"] = template["citations"].replace(" | ", "\n\n")
    template["study_executive_names"] = credits["Names"]["Study Executive"]
    template["local_collaborators_names"] = credits["Names"][city]
    if str(template["translation_names"]) == "nan":
        template["translation"] = ""
        template["translation_names"] = ""

    template["suggested_citation"] = "{}: {}".format(
        phrases["citation_word"],
        phrases["citation_doi"].format(
            city=city, country=country, language=vernacular
        ),
    )
    template["licence_image"] = "logos/by-nc.jpg"
    template.render()

    # Output scorecard pdf
    pdf.output(
        f"{scorecard_path}/{city_name} - {languages.loc[languages['name']=='title_series_line1',language].values[0].replace(':','')} - GHSCIC 2022 - {vernacular}.pdf"
    )
    return f"Scorecard generated: {scorecard_path}/scorecard_{city}.pdf"
