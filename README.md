# Geiger

Currently, clustering relies on my implementation of IHAC in [galaxy](https://github.com/ftzeng/galaxy) which is still kind of finicky.


## Usage

Install requirements:

    $ pip install -r requirements.txt

Setup the config as necessary:

    $ cp config-sample.py config.py; vi config.py

Train the vectorizer:

    $ python main.py train

Run the server:

    $ python main.py server

Then try out a NYT article:

    localhost:5001/geiger/<NYT article url>


## To Do

- build sentiment features
- build subjectivity features
- build named entity or keyword features
- how to evaluate?
- how to determine a good distance cutoff?
- how to determine a good minimum cluster size?
