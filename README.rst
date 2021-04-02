Utilities
=========

.. image:: https://travis-ci.org/haaksmash/pyutils.svg?branch=master
    :target: https://travis-ci.org/haaksmash/pyutils

In any project we usually need a set of functions that we use again and again and that don't typically belong to the project.
They go in an util file because they do not belong to the specific application logic.
We also spend time in every project searching, adapting and copy/pasting these functions.

In this project, I, decided to collect those functions and put them in one centralized place!

Functionalities
+++++++++++++++

Utilities libprary is broken up into broad domains of functionality in orther to easily remember where thinks are located.

Dictionnaries
-------------

DotDict
-------

enum
----

Python doesn't have a built-in way to define an enum, so this module provides (what I think) is a pretty clean way to go about them.

.. code-block:: python

    from utils import enum

    class ModelTypes(enum.Enum):
        CLASSIFICATION = 0
        REGRESSION = 1

        # Defining an Enum class allows you to specify a few
        # things about the way it's going to behave.
        class Options:
            frozen = True # can't change attributes
            strict = True # can only compare to itself; i.e., Colors.RED == Animals.COW
                          # will raise an exception.

Once defined, use is straightforward:

.. code-block:: python

    >>> ModelTypes



dicts
-----

intersections, differences, winnowing, a few specialized dicts...

lists
-----

flatten and unlisting. also ``flat_map``!

bools
-----

currently only provides an ``xor`` function.

dates
-----


objects
-------

provides ``get_attr``, which is really just a convenient way to do deep ``getattr`` chaining:

.. code-block:: python

    >>> get_attr(complicated, 'this.is.a.deep.string', default=None)
    "the deep string"  # or None, if anything in the lookup chain didn't exist

There's also an ``immutable`` utility, which will wrap an object and preven all attribute changes,
recursively by default. Any attempt to set attributes on the wrapped object will raise an ``AttributeError``:

.. code-block:: python

    >>> imm = immutable(something)
    >>> imm
    <Immutable Something: <Something>>
    >>> imm.red
    <Immutable SomethingElse: <SomethingElse: red>>
    >>> imm.red = SomethingElse('blue')
    # ...
    AttributeError: This object has been marked as immutable; you cannot set its attributes.
    >>> something.red = SomethingElse('blue')
    >>> imm.red
    <Immutable SomethingElse: <SomethingElse: blue>>

You can toggle the recursive immutability by specifying the 'recursive' flag.

Installation (via pip)
++++++++++++++++++++++

    pip install kdmt

