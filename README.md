# triplet_exp
## Experiment 1:
Train deep-nets on xor data (fixed sample size) with kaleab's architectures, take off the penultimate layer, train the penultimate layer for r-xor for varying sample size, see the accuracy, compare it to the accuracy achieved when the whole network was trained on r-xor.

Install from Github
-------------------
You can manually download ``triplet`` by cloning the git repo master version and
running the ``setup.py`` file. That is, unzip the compressed package folder
and run the following from the top-level source directory using the Terminal::

    $ git clone https://github.com/neurodata/triplet_exp.git
    $ cd triplet_exp
    $ python3 setup.py install

