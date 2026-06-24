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

   * - :py:obj:`_updater_reject_reason <deepxube.factories.updater_factory._updater_reject_reason>`
     - .. autodoc2-docstring:: deepxube.factories.updater_factory._updater_reject_reason
          :summary:
   * - :py:obj:`get_updater <deepxube.factories.updater_factory.get_updater>`
     - .. autodoc2-docstring:: deepxube.factories.updater_factory.get_updater
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

.. py:function:: _updater_reject_reason(up_cls: typing.Type[deepxube.base.updater.Update], domain: deepxube.base.domain.Domain, pathfind_t: typing.Type[deepxube.base.pathfinding.PathFind], her: bool, func_update: str) -> typing.Optional[str]
   :canonical: deepxube.factories.updater_factory._updater_reject_reason

   .. autodoc2-docstring:: deepxube.factories.updater_factory._updater_reject_reason

.. py:function:: get_updater(domain: deepxube.base.domain.Domain, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs, her: bool, func_update: str) -> deepxube.base.updater.Update
   :canonical: deepxube.factories.updater_factory.get_updater

   .. autodoc2-docstring:: deepxube.factories.updater_factory.get_updater
