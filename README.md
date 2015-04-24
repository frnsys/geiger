# Geiger

(work in progress - for details, see the [proposal](proposal/proposal.md).)

## Usage

Install requirements:

    $ pip install -r requirements.txt

Setup the config as necessary:

    $ cp config-sample.py config.py; vi config.py

Download the necessary corpora:

    $ python -m textblob.download_corpora

For the `baseline` ("talked about") strategy, the following must be trained:

    $ python train_phrases.py
    $ python train_idf.py

You can use any corpus to train these on; I used the body text of about 120k NYT articles and it has worked well. The more, the better, most likely.

These are used to better identify phrases in text and to have some notion of salience (inverse document frequency).

Run the server:

    $ python run.py server

You can evaluate the clustering algos with the currently selected featurizers on a (very) small labeled dataset:

    $ python run.py eval

Then try out the demo:

    localhost:5001/

This will output the selections from a variety of different strategies.

To visualize the output of a clustering strategy (for debugging/tweaking purposes):

    localhost:5001/visualize/[lda, hac, k_means, dbscan]
    localhost:5001/visualize/[lda, hac, k_means, dbscan]/<NYT article url>

To see the results of the baseline ("talked about") algorithm:

    localhost:5001/talked-about
    localhost:5001/talked-about/<NYT article url>

### Other notes

If you are using the `semsim` strategy, you will need to train a Word2Vec model and then run the `w2v.py` process:

    $ python w2v.py

The Word2Vec model is quite big and takes some time to load. The `w2v.py` script will run it as an independent process with a listener, which the `semsim` strategy interfaces with.

If you are using the `polisent` featurizer, you must train the `polisent` model:

    $ python run.py train_polisent


## Configuration

The main configuration option is `config.features` in which you specify what featurizers to use.
Clustering strategies other than LDA use these features to generate clusters.
