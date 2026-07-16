:py:mod:`deepxube.factories.updater_factory`
============================================

.. py:module:: deepxube.factories.updater_factory

.. autodoc2-docstring:: deepxube.factories.updater_factory
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_domain_compat_updater_names <deepxube.factories.updater_factory.get_domain_compat_updater_names>`
     - .. autodoc2-docstring:: deepxube.factories.updater_factory.get_domain_compat_updater_names
          :summary:
   * - :py:obj:`get_pathfind_compat_updater_names <deepxube.factories.updater_factory.get_pathfind_compat_updater_names>`
     - .. autodoc2-docstring:: deepxube.factories.updater_factory.get_pathfind_compat_updater_names
          :summary:
   * - :py:obj:`get_updater_from_args <deepxube.factories.updater_factory.get_updater_from_args>`
     - .. autodoc2-docstring:: deepxube.factories.updater_factory.get_updater_from_args
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`updater_factory <deepxube.factories.updater_factory.updater_factory>`
     - .. autodoc2-docstring:: deepxube.factories.updater_factory.updater_factory
          :summary:

API
~~~

.. py:data:: updater_factory
   :canonical: deepxube.factories.updater_factory.updater_factory
   :type: deepxube.base.factory.Factory[deepxube.base.updater.Update]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.updater_factory.updater_factory

.. py:function:: get_domain_compat_updater_names(domain_t: typing.Type[deepxube.base.domain.Domain]) -> typing.List[str]
   :canonical: deepxube.factories.updater_factory.get_domain_compat_updater_names

   .. autodoc2-docstring:: deepxube.factories.updater_factory.get_domain_compat_updater_names

.. py:function:: get_pathfind_compat_updater_names(pathfind_t: typing.Type[deepxube.base.pathfinding.PathFind]) -> typing.List[str]
   :canonical: deepxube.factories.updater_factory.get_pathfind_compat_updater_names

   .. autodoc2-docstring:: deepxube.factories.updater_factory.get_pathfind_compat_updater_names

.. py:function:: get_updater_from_args(domain: deepxube.base.domain.Domain, pathfind: deepxube.base.pathfinding.PathFind, pathfind_name_args: str, updater_fns: deepxube.base.pathfind_fns.UFNs, updater_name_args: str) -> typing.Tuple[deepxube.base.updater.Update, str]
   :canonical: deepxube.factories.updater_factory.get_updater_from_args

   .. autodoc2-docstring:: deepxube.factories.updater_factory.get_updater_from_args
