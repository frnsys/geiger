# Geiger

(work in progress)

Currently, the IHAC clustering strategy relies on my implementation of IHAC in [galaxy](https://github.com/ftzeng/galaxy) which is still kind of finicky.


## Usage

Install requirements:

    $ pip install -r requirements.txt

Setup the config as necessary:

    $ cp config-sample.py config.py; vi config.py

Run the server:

    $ python main.py server

Then try out a NYT article:

    localhost:5001/geiger/<NYT article url>

To visualize the output of a clustering strategy (for debugging/tweaking purposes):

    localhost:5001/visualize_strat/[lda, hac, ihac, k_means]
    localhost:5001/visualize_strat/[lda, hac, ihac, k_means]/<NYT article url>


## Method

The general approach is:

- Identify comment clusters
- For clusters satisfying some criteria (e.g. minimum size), select a representative sentence.


## Configuration

The main configuration option is `config.features` in which you specify what featurizers to use.
Clustering strategies other than LDA use these features to generate clusters.


## To Do

- clean up
- figure out how to evaluate/compare approaches
- other ways of filtering out sentences? In general, look for cues which reference context beyond the sentence itself. For example:
    - non-"I" pronouns
    - starting with terms like "However", "For example", etc
