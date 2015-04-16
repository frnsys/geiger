# Geiger

(work in progress - for details, see the [proposal](proposal/proposal.md).)

The IHAC clustering strategy relies on my implementation of IHAC in [galaxy](https://github.com/ftzeng/galaxy) which is still kind of finicky.


## Usage

Install requirements:

    $ pip install -r requirements.txt

Setup the config as necessary:

    $ cp config-sample.py config.py; vi config.py

If you need to, train the necessary models (i.e. those which power other featurizers).
For example, the `polisent` featurizer must have its `polisent` model trained:

    $ python run.py train_polisent

Run the server:

    $ python run.py server

You can evaluate the clustering algos with the currently selected featurizers on a (very) small labeled dataset:

    $ python run.py eval

Then try out the demo:

    localhost:5001/

This will output the selections from a variety of different strategies.

To visualize the output of a clustering strategy (for debugging/tweaking purposes):

    localhost:5001/visualize_strat/[lda, hac, ihac, k_means, dbscan]
    localhost:5001/visualize_strat/[lda, hac, ihac, k_means, dbscan]/<NYT article url>


### Recommendations

(note: the following isn't quite ready)

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
- write a chrome extension to mock the selections on a live NYT page
