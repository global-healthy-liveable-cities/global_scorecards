"""
Global scorecards.

Format and save indicator reports.
"""
import argparse
import os
import sys

import pandas as pd

# import and set up functions
import _data_setup
import _report_functions
from batlow import batlow_map

sys.path.insert(0, "../../process/pre_process")
from _project_setup import *

# Set up commandline input parsing
parser = argparse.ArgumentParser(
    description="Scorecards for the Global Healthy and Sustainable Cities Indicator Collaboration Study"
)

parser.add_argument(
    "--cities",
    default="Maiduguri,Mexico City,Baltimore,Phoenix,Seattle,Sao Paulo,Hong Kong,Chennai,Bangkok,Hanoi,Graz,Ghent,Bern,Olomouc,Cologne,Odense,Barcelona,Valencia,Vic,Belfast,Lisbon,Adelaide,Melbourne,Sydney,Auckland",
    help=(
        "A list of cities, for example: Baltimore, Phoenix, Seattle, Adelaide, Melbourne, "
        "Sydney, Auckland, Bern, Odense, Graz, Cologne, Ghent, Belfast, Barcelona, Valencia, Vic, "
        "Lisbon, Olomouc, Hong, Kong, Mexico City, Sao, Paulo, Bangkok, Hanoi, Maiduguri, Chennai"
    ),
)

parser.add_argument(
    "--generate_resources",
    action="store_true",
    default=False,
    help="Generate images from input data for each city? Default is False.",
)

parser.add_argument(
    "--language",
    default="English",
    type=str,
    help="The desired language for presentation, as defined in the template workbook languages sheet.",
)

parser.add_argument(
    "--auto_language",
    action="store_true",
    default=False,
    help="Identify all languages associated with specified cities and prepare reports for these.",
)

parser.add_argument(
    "--by_city",
    action="store_true",
    default=True,
    help="Save reports in city-specific sub-folders.",
)

parser.add_argument(
    "--by_language",
    action="store_true",
    default=True,
    help="Save reports in language-specific sub-folders (default).",
)

parser.add_argument(
    "--templates",
    default="web",
    help=(
        'A list of templates to iterate outputs over, for example: "web" (default), or "web,print"\n'
        'The words listed correspond to sheets present in the configuration file, prefixed by "template_",'
        'for example, "template_web" and "template_print".  These files contain the PDF template layout '
        "information required by fpdf2 for pagination of the output PDF files."
    ),
)

parser.add_argument(
    "--configuration",
    default="_report_configuration.xlsx",
    help=(
        "An XLSX workbook containing spreadsheets detailing template layout(s), prose, fonts and city details "
        "to be drawn upon when formatting reports."
    ),
)


config = parser.parse_args()
templates = [f"template_{x.strip()}" for x in config.templates.split(",")]
cmap = batlow_map


if __name__ == "__main__":
    # Run all specified language-city permutations if auto-language detection
    languages = _report_functions.get_and_setup_language_cities(config)

    for language in languages.index:
        cities = languages[language]
        _report_functions.generate_report_for_language(
            config, language, cities, indicators
        )
