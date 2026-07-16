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

   * - :py:obj:`get_domain_compat_pathfind_names <deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names>`
     - .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names
          :summary:
   * - :py:obj:`get_pathfind_name_kwargs <deepxube.factories.pathfinding_factory.get_pathfind_name_kwargs>`
     - .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_pathfind_name_kwargs
          :summary:
   * - :py:obj:`get_pathfind_from_arg <deepxube.factories.pathfinding_factory.get_pathfind_from_arg>`
     - .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_pathfind_from_arg
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

.. py:function:: get_domain_compat_pathfind_names(domain_t: typing.Type[deepxube.base.domain.Domain]) -> typing.List[str]
   :canonical: deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names

   .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_domain_compat_pathfind_names

.. py:function:: get_pathfind_name_kwargs(pathfind: str) -> typing.Tuple[str, typing.Dict[str, typing.Any]]
   :canonical: deepxube.factories.pathfinding_factory.get_pathfind_name_kwargs

   .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_pathfind_name_kwargs

.. py:function:: get_pathfind_from_arg(domain: deepxube.base.domain.Domain, pathfind_fns: deepxube.base.pathfind_fns.PFNs, pathfind_name_args: str) -> typing.Tuple[deepxube.base.pathfinding.PathFind, str, str]
   :canonical: deepxube.factories.pathfinding_factory.get_pathfind_from_arg

   .. autodoc2-docstring:: deepxube.factories.pathfinding_factory.get_pathfind_from_arg
