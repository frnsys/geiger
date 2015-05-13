# Geiger

(work in progress - for details, see the [proposal](proposal/proposal.md).)

## Usage

Install requirements:

    $ pip install -r requirements.txt

Setup the config as necessary:

    $ cp config-sample.py config.py; vi config.py

Download the necessary corpora:

    $ python -m textblob.download_corpora

You will need to prepare some data:

    $ python prep.py train_phrases
    $ python prep.py train_idf

You can use any corpus to train these on; I used the body text of about 120k NYT articles and it has worked well. The more, the better, most likely.

These are used to better identify phrases in text and to have some notion of salience (inverse document frequency).

Run the server:

    $ python server.py

Then try out the demo:

    localhost:5001/

To see the results of the baseline ("talked about") algorithm:

    localhost:5001/talked-about
    localhost:5001/talked-about/<NYT article url>


## Development

If you are developing and need to reload Geiger a lot, you are in for a bad time. The phrase and Word2Vec models take a very long time to load.

Fortunately, things are setup so that you can run each of these in their own separate process, which don't need to be reloaded.
If you set `remote=True` in `config.py`, the functions which rely on the phrase and Word2Vec models will call out to these
separate processes instead of loading the models directly.

The downside is that calling out to separate processes like this slows the usage of these models, but you'll likely be saving time overall.
