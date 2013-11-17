SKLEARN - TWSS
==============

This is an implementation of a simple double meaning classifier in Python. 

This currently uses a Naive Bayes classifier (the SKLEARN implementation) as a
Python package. This was inspired by the `twss
<https://github.com/sengupta/twss>`_ and uses the same data corpus. 

Difference from Repo forked from
--------------------------------

1. Completely written from scratch using `SCIKIT-LEARN` as I wanted to learn more about sklearn and naive bayes classifier.
2. This has a `cross_validate()` function which gives a score of 0.931 for now.

Suggestions welcome. Do file bugs. Fork away. Send me pull requests. 

Setup Instructions
------------------

.. code-block:: bash
 
    $ pip install -r requirements.txt

Demo
----

Once this is installed, you can take it out for a spin: 

.. code-block:: python 

    In [1]: from twss import TWSS

    In [2]: x = TWSS()

    In [3]: x.cross_validate()
    0.934612651031

    In [10]: x("Boka has a hard one.")
    [ True]

    In [12]: x("Boka is awesome.")
    [False]

The first call can take a while- the module needs to train the classifier
against the pre-installed training dataset. 

Getting dirty
-------------

You can supply your own training data using positive and negative corpus files: 

.. code-block:: python 

    In [13]: x = TWSS(positive_corpus_file='foo.txt', negative_corpus_file='bar.txt')

Roadmap
-------

- Making this pip-installable.
- Writing a sample web app.
- Writing a sample Twitter client.