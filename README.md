# Geiger

(work in progress)

The IHAC clustering strategy relies on my implementation of IHAC in [galaxy](https://github.com/ftzeng/galaxy) which is still kind of finicky.


## Usage

Install requirements:

    $ pip install -r requirements.txt

Setup the config as necessary:

    $ cp config-sample.py config.py; vi config.py

Run the server:

    $ python main.py server

Then try out the demo:

    localhost:5001/

This will output the selections from a variety of different strategies.

To visualize the output of a clustering strategy (for debugging/tweaking purposes):

    localhost:5001/visualize_strat/[lda, hac, ihac, k_means]
    localhost:5001/visualize_strat/[lda, hac, ihac, k_means]/<NYT article url>


### Recommendations

It's recommended that you also run the `Doc2Vec` process:

    $ python doc2vec.py

The Doc2Vec model I'm using is huge and takes a really long time to load. The `doc2vec.py` script will run it as an independent process with a listener.


## Method

The general approach is:

- Identify comment clusters
- For clusters satisfying some criteria (e.g. minimum size), select a representative sentence.


## Configuration

The main configuration option is `config.features` in which you specify what featurizers to use.
Clustering strategies other than LDA use these features to generate clusters.


## To Do

- figure out how to evaluate/compare approaches
- other ways of filtering out sentences? In general, look for cues which reference context beyond the sentence itself. For example:
    - non-"I" pronouns
    - starting with terms like "However", "For example", "(", "So", etc
    - does not start and end with quotes
- write a chrome extension to mock the selections on a live NYT page
