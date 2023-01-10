# Scorecards for the Global Healthy and Sustainable Cities Indicator Collaboration Study

The code in this repository draws on results from the [Global Healthy and Sustainable Cities Indicator Collaboration Study](https://github.com/global-healthy-liveable-cities/global-indicators) to generate policy 'scorecard' reports.

The respository is located as a subfolder of the study's spatial components repository, located within the [analysis](https://github.com/global-healthy-liveable-cities/global-indicators/tree/main/analysis) folder.

The version of the code tagged '0.1' was used to prepare the reports for our 25 city analysis (https://doi.org/10.25439/rmt.c.6012649), iterating over the processed set of cities and producing reports in multiple languages and associated fonts as configured.  The current version of the code is designed to support the Global Healthy and Sustinable City Indicators Collaboration [1000 Cities challenge](https://www.healthysustainablecities.org/1000cities), supporting generation of reports for newly processed cities in conjunction with supplied policy review results.  Configuration of reporting ties in with the [indicators.yml](https://github.com/global-healthy-liveable-cities/global-indicators/blob/enhancements/process/configuration/templates/indicators.yml) and [policies.yml](https://github.com/global-healthy-liveable-cities/global-indicators/blob/enhancements/process/configuration/templates/policies.yml) files which are to be stored with optional modification in a project's [configuration folder](https://github.com/global-healthy-liveable-cities/global-indicators/tree/enhancements/process/configuration).

The code is run using the Docker image for our [Global Healthy and Sustainable Cities Indicator Collaboration Study](https://github.com/global-healthy-liveable-cities/global-indicators), and launched from the analysis/global_scorecards directory:

```
ghsci@docker-desktop:~/work/analysis/global_scorecards$ python _generate_reports.py --help
Global Healthy Liveable Cities Indicator Study Collaboration, version 1.2

Generate reports

usage: _generate_reports.py [-h] [--city CITY] [--generate_resources] [--language LANGUAGE] [--auto_language]
                            [--templates TEMPLATES [TEMPLATES ...]] [--configuration CONFIGURATION]

Reports and infographic scorecards for the Global Healthy and Sustainable City Indicators Collaboration

optional arguments:
  -h, --help            show this help message and exit
  --city CITY           The city for which reports are to be generated.
  --generate_resources  Generate images from input data for each city? Default is True.
  --language LANGUAGE   The desired language for presentation, as defined in the template workbook languages sheet.
  --auto_language       Identify all languages associated with specified cities and prepare reports for these.
  --templates TEMPLATES [TEMPLATES ...]
                        A list of templates to iterate outputs over, for example: "web" (default), or "web,print" The words
                        listed correspond to sheets present in the configuration file, prefixed by "template_",for example,
                        "template_web" and "template_print". These files contain the PDF template layout information required by
                        fpdf2 for pagination of the output PDF files.
  --configuration CONFIGURATION
                        An XLSX workbook containing spreadsheets detailing template layout(s), prose, fonts and city details to
                        be drawn upon when formatting reports.
```

So, given available translations have been defined, the follow would generate scorecards for Barcelona, Valencia, Vic and Mexico City in Spanish:
```
python _generate_reports.py --city ghent_v2 --auto_language
```

The code identifies required data sources, and extracts key information which is passed to functions in scorecard_functions.py to generate resources, format pages, and output the final PDF layout, according to the locations, pages and prose specified in the _report_configuration.xlsx file [^1].   

Carl Higgs
carl.higgs@rmit.edu.au

[^1]: The _report_configuration.xlsx file makes use of relative formulas for placement of nested elements.  This makes it easier to quickly move entire blocks of content around the page when fine-tuning the template, compared with a plain text format like CSV.  It also contains multiple worksheets, used to define phrases in different languages and their associated font type faces.
