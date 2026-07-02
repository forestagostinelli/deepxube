:py:mod:`deepxube.updaters.updater_sup`
=======================================

.. py:module:: deepxube.updaters.updater_sup

.. autodoc2-docstring:: deepxube.updaters.updater_sup
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpdateHeurVSup <deepxube.updaters.updater_sup.UpdateHeurVSup>`
     -
   * - :py:obj:`UpdateHeurQSup <deepxube.updaters.updater_sup.UpdateHeurQSup>`
     -
   * - :py:obj:`UpdatePolicySup <deepxube.updaters.updater_sup.UpdatePolicySup>`
     -

API
~~~

.. py:class:: UpdateHeurVSup(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurV`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindNodeSup`\ ], :py:obj:`deepxube.base.updater.UpdateSup`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindNodeSup`\ , :py:obj:`deepxube.base.pathfinding.InstanceNode`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurVSup.domain_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.pathfinding.supervised.PathFindNodeSup]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurVSup.pathfind_type

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.InstanceNode], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurVSup._get_instance_data_norb

.. py:class:: UpdateHeurQSup(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurQ`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSup`\ ], :py:obj:`deepxube.base.updater.UpdateSup`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSup`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurQSup.domain_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.pathfinding.supervised.PathFindEdgeSup]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurQSup.pathfind_type

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.InstanceEdge], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurQSup._get_instance_data_norb

.. py:class:: UpdatePolicySup(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.updaters.updater_sup.UpdatePolicySup

   Bases: :py:obj:`deepxube.base.updater.UpdatePolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSamp`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ ], :py:obj:`deepxube.base.updater.UpdateSup`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSamp`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_sup.UpdatePolicySup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdatePolicySup.domain_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.pathfinding.supervised.PathFindEdgeSamp]
      :canonical: deepxube.updaters.updater_sup.UpdatePolicySup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdatePolicySup.pathfind_type

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.InstanceEdge], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_sup.UpdatePolicySup._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdatePolicySup._get_instance_data_norb
