====================================
Scipion plugin for Relion Tomography
====================================

.. image:: https://img.shields.io/pypi/v/scipion-em-gapstop.svg
        :target: https://pypi.python.org/pypi/scipion-em-gapstop
        :alt: PyPI release

.. image:: https://img.shields.io/pypi/l/scipion-em-gapstop.svg
        :target: https://pypi.python.org/pypi/scipion-em-gapstop
        :alt: License

.. image:: https://img.shields.io/pypi/pyversions/scipion-em-gapstop.svg
        :target: https://pypi.python.org/pypi/scipion-em-gapstop
        :alt: Supported Python versions

.. image:: https://img.shields.io/pypi/dm/scipion-em-gapstop
        :target: https://pypi.python.org/pypi/scipion-em-gapstop
        :alt: Downloads

This plugin provide a wrapper around the program `GAPSTOP_TM <https://bturo.pages.mpcdf.de/gapstop_tm/index.html>`_
(GPU-Accelerated Python STOPgap for Template Matching). to use it within
`Scipion <https://scipion-em.github.io/docs/release-3.0.0/index.html>`_ framework.

Installation
------------

You will need to use `3.0 <https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html>`_
version of Scipion to be able to run these protocols. To install the plugin, you have two options:


a) Stable version:

.. code-block::

    scipion3 installp -p scipion-em-fidder

b) Developer's version

    * download the repository from github:

    .. code-block::

        git clone -b devel https://github.com/scipion-em/scipion-em-fidder.git

    * install:

    .. code-block::

        scipion3 installp -p /path/to/scipion-em-fidder --devel

To check the installation, simply run the following Scipion test for the plugin:

    .. code-block::

        scipion3 tests fidder.tests.tests_fidder.TestFidder

Protocols
-----------

TBD

Latest plugin versions
----------------------

If you want to check the latest version and release history go to `CHANGES <https://github.com/scipion-em-reliotomo/gapstop/blob/master/CHANGES.txt>`_


References
----------

1. Cruz-León, S., Majtner, T., Hoffmann, P.C. et al. Nat Commun 15, 3992 (2024).


