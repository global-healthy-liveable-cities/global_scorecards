# Scorecards for the Global Healthy and Sustainable Cities Indicator Collaboration Study

The code in this repository draws on results from the [Global Healthy and Sustainable Cities Indicator Collaboration Study](https://github.com/global-healthy-liveable-cities/global-indicators) to generate policy 'scorecard' reports.

The respository is ideally located as a subfolder of the study's spatial components repository, located within the [analysis](https://github.com/global-healthy-liveable-cities/global-indicators/tree/main/analysis) folder.

The input data required are described in the global_scorecards.py file; this data may be awaiting publication to be made public, and is not located in this nor the spatial repository at this stage:

```
gpkg_hexes = os.path.abspath('../../process/data/output/global_indicators_hex_250m_2021-06-21.gpkg')
csv_city_indicators = os.path.abspath("../../process/data/output/global_indicators_city_2021-06-21.csv")
csv_hex_indicators = os.path.abspath("../../process/data/output/global_indicators_hex_250m_2021-06-21.csv")
csv_thresholds_data = os.path.abspath("data/Global Indicators 2020 - thresholds summary estimates.csv")
xlsx_policy_data = os.path.abspath("data/Policy Figures 1 & 2_23 Dec_numerical.xlsx")
xlsx_scorecard_template = 'scorecard_template_elements.xlsx'
```

Python modules required for running the code include pandas, geopandas, and fpdf2 (for PDF generation).  A Dockerfile for building a Docker environment including the required modules to run the project (and more [^1]) are provided in the Docker folder.  The Docker image can be built from within the docker folder by executing `docker build -t carlhiggs/global_scorecards .`, and then from the base project directory of [Global Healthy and Sustainable Cities Indicator Collaboration Study](https://github.com/global-healthy-liveable-cities/global-indicators) running either of the following from command line:

```{Linux}
docker run --rm -it --shm-size=2g --net=host -v "$PWD":/home/ghsi/work carlhiggs/global_scorecards /bin/bash
```

```{Windows}
docker run --rm -it --shm-size=2g --net=host -v "%cd%":/home/ghsi/work carlhiggs/global_scorecards /bin/bash
```

With the required modules installed (or run via the supplied Docker image) and data resources available relative to the project directory in the paths listed above, the scorecards can be generated by running the global_scorecards.py Python script.  
```
python .\global_scorecards.py --help
usage: global_scorecards.py [-h] [--cities CITIES] [--generate_resources] [--language LANGUAGE] [--auto_language]

Scorecards for the Global Healthy and Sustainable Cities Indicator Collaboration Study

optional arguments:
  -h, --help            show this help message and exit
  --cities CITIES       A list of cities, for example: Baltimore, Phoenix, Seattle, Adelaide, Melbourne, Sydney,
                        Auckland, Bern, Odense, Graz, Cologne, Ghent, Belfast, Barcelona, Valencia, Vic, Lisbon,
                        Olomouc, Hong, Kong, Mexico City, Sao, Paulo, Bangkok, Hanoi, Maiduguri, Chennai
  --generate_resources  Generate images from input data for each city? Default is False.
  --language LANGUAGE   The desired language for presentation, as defined in the template workbook languages sheet.
  --auto_language       Identify all languages associated with specified cities and prepare reports for these.
```

So, given available translations have been defined, the follow would generate scorecards for Barcelona, Valencia, Vic and Mexico City in Spanish:
```
python global_scorecards.py --cities "Barcelona, Valencia, Vic, Mexico City" --language Spanish --generate_resources
```

Alternately, reports for all cities can be produced for all aligned language translations:

```
python global_scorecards.py --auto_language --generate_resources
```

If image resources have already been generated, and only the layout has been modified, the `--generate_resources` argument may be omitted.

You can also generate reports for all cities aligned with one particular language, e.g.

```
python global_scorecards.py --auto_language --language "Catalan"
```

Or generate reports for all languages aligned with a subset of cities:

```
python global_scorecards.py --auto_language --cities "Barcelona, Valencia, Vic, Mexico City"
```

The code identifies required data sources, and extracts key information which is passed to functions in scorecard_functions.py to generate resources, format pages, and output the final PDF layout, according to the locations, pages and prose specified in the scorecard_template_elements.xlsx file [^2].   

Carl Higgs
carl.higgs@rmit.edu.au

[^1]: The Docker image also contains additional dependencies not required for this project, including OSMnx.  This is because it serves as a proof of concept refactoring of the image for the broader [Global Healthy and Sustainable Cities Indicator Collaboration Study](https://github.com/global-healthy-liveable-cities/global-indicators) Docker image.
[^2]: The scorecard_template_elements.xlsx file makes use of relative formulas for placement of nested elements.  This makes it easier to quickly move entire blocks of content around the page when fine-tuning the template, compared with a plain text format like CSV.  It also contains multiple worksheets, used to define phrases in different languages and their associated font type faces.
