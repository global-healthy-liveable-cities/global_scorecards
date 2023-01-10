"""
Global scorecards.

Format and save indicator reports.
"""
import argparse
import os
import sys

# import and set up functions
import _report_functions

sys.path.insert(0, "../../process/pre_process")
from _project_setup import indicators, locale, policies, regions

# Set up commandline input parsing
parser = argparse.ArgumentParser(
    description="Reports and infographic scorecards for the Global Healthy and Sustainable City Indicators Collaboration"
)

parser.add_argument(
    "--city",
    default=locale,
    type=str,
    help="The city for which reports are to be generated.",
)

parser.add_argument(
    "--generate_resources",
    action="store_true",
    default=True,
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
    "--templates",
    nargs="+",
    default=["web"],
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
config.city_path = regions[config.city]["locale_dir"]
if not os.path.exists(config.city_path):
    sys.exit(
        f"\n\nProcessed resource folder for this city couldn't be located:"
        f"\n[{config.city_path}]"
        "\nPlease ensure city has been successfully processed before continuing\n"
    )

# # for debugging only
# languages = _report_functions.get_and_setup_language_cities(config)
# language = 'English'

# _report_functions.generate_report_for_language(
# config,
# language,
# indicators,
# regions,
# policies
# )


def main():
    languages = _report_functions.get_and_setup_language_cities(config)
    for language in languages:
        # print(f"\{language} language reports:")
        _report_functions.generate_report_for_language(
            config, language, indicators, regions, policies
        )


if __name__ == "__main__":
    main()
