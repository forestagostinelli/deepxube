:py:mod:`deepxube.factories.pathfinding_factory`
================================================

.. py:module:: deepxube.factories.pathfinding_factory

.. autodoc2-docstring:: deepxube.factories.pathfinding_factory
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_pathfind_functions <deepxube.factories.pathfinding_factory.get_pathfind_functions>`
     - .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_pathfind_functions
          :summary:
   * - :py:obj:`get_domain_compat_pathfind_names <deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names>`
     - .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`pathfinding_factory <deepxube.factories.pathfinding_factory.pathfinding_factory>`
     - .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.pathfinding_factory
          :summary:

API
~~~

.. py:data:: pathfinding_factory
   :canonical: deepxube.factories.pathfinding_factory.pathfinding_factory
   :type: deepxube.base.factory.Factory[deepxube.base.pathfinding.PathFind]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.pathfinding_factory

.. py:function:: get_pathfind_functions(pathfind_name: str, heur_fn: typing.Optional[deepxube.base.heuristic.HeurFn], policy_fn: typing.Optional[deepxube.base.heuristic.PolicyFn]) -> typing.Any
   :canonical: deepxube.factories.pathfinding_factory.get_pathfind_functions

   .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_pathfind_functions

.. py:function:: get_domain_compat_pathfind_names(domain_t: typing.Type[deepxube.base.domain.Domain]) -> typing.List[str]
   :canonical: deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names

   .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names
