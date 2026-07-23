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
   * - :py:obj:`UpdateVSup <deepxube.updaters.updater_sup.UpdateVSup>`
     -
   * - :py:obj:`UpdateQSup <deepxube.updaters.updater_sup.UpdateQSup>`
     -
   * - :py:obj:`UpdatePSup <deepxube.updaters.updater_sup.UpdatePSup>`
     -

API
~~~

.. py:class:: UpdateHeurVSup(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurV`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindNodeSup`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ], :py:obj:`deepxube.base.updater.UpdateSup`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindNodeSup`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurVSup.domain_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.pathfinding.supervised.PathFindNodeSup]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurVSup.pathfind_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurV]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurVSup.updater_functions_type

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurVSup._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurVSup._get_instance_data_norb

.. py:class:: UpdateHeurQSup(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup

   Bases: :py:obj:`deepxube.base.updater.UpdateHeurQ`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSup`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`\ ], :py:obj:`deepxube.base.updater.UpdateSup`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSup`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurQSup.domain_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.pathfinding.supervised.PathFindEdgeSup]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurQSup.pathfind_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsHeurQ]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurQSup.updater_functions_type

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_sup.UpdateHeurQSup._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdateHeurQSup._get_instance_data_norb

.. py:class:: UpdatePolicySup(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.updaters.updater_sup.UpdatePolicySup

   Bases: :py:obj:`deepxube.base.updater.UpdatePolicy`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSamp`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`\ ], :py:obj:`deepxube.base.updater.UpdateSup`\ [\ :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.pathfinding.supervised.PathFindEdgeSamp`\ , :py:obj:`deepxube.base.pathfinding.Instance`\ , :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.updaters.updater_sup.UpdatePolicySup.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdatePolicySup.domain_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.pathfinding.supervised.PathFindEdgeSamp]
      :canonical: deepxube.updaters.updater_sup.UpdatePolicySup.pathfind_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdatePolicySup.pathfind_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.pathfind_fns.UFNsPolicy]
      :canonical: deepxube.updaters.updater_sup.UpdatePolicySup.updater_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdatePolicySup.updater_functions_type

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.pathfinding.Instance], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.updaters.updater_sup.UpdatePolicySup._get_instance_data_norb

      .. autodoc2-docstring:: deepxube.updaters.updater_sup.UpdatePolicySup._get_instance_data_norb

.. py:class:: UpdateVSup()
   :canonical: deepxube.updaters.updater_sup.UpdateVSup

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`

.. py:class:: UpdateQSup()
   :canonical: deepxube.updaters.updater_sup.UpdateQSup

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`

.. py:class:: UpdatePSup()
   :canonical: deepxube.updaters.updater_sup.UpdatePSup

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`
