# Geiger

(work in progress)

Currently, clustering relies on my implementation of IHAC in [galaxy](https://github.com/ftzeng/galaxy) which is still kind of finicky.


## Usage

Install requirements:

    $ pip install -r requirements.txt

Setup the config as necessary:

    $ cp config-sample.py config.py; vi config.py

Run the server:

    $ python main.py server

Then try out a NYT article:

    localhost:5001/geiger/<NYT article url>

To visualize and explore the clustering output (for debugging/tweaking purposes):

    localhost:5001/visualize/<NYT article url>


## To Do

- try regular agglomerative clustering and K-means
- how to select sentences?
