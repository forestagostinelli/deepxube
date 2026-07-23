:py:mod:`deepxube.base.updater`
===============================

.. py:module:: deepxube.base.updater

.. autodoc2-docstring:: deepxube.base.updater
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UpArgs <deepxube.base.updater.UpArgs>`
     - .. autodoc2-docstring:: deepxube.base.updater.UpArgs
          :summary:
   * - :py:obj:`Update <deepxube.base.updater.Update>`
     -
   * - :py:obj:`UpdateHER <deepxube.base.updater.UpdateHER>`
     -
   * - :py:obj:`UpdateHasHeurV <deepxube.base.updater.UpdateHasHeurV>`
     -
   * - :py:obj:`UpdateHasHeurQ <deepxube.base.updater.UpdateHasHeurQ>`
     -
   * - :py:obj:`UpdateHasPolicy <deepxube.base.updater.UpdateHasPolicy>`
     -
   * - :py:obj:`UpdateSup <deepxube.base.updater.UpdateSup>`
     -
   * - :py:obj:`UpRLArgs <deepxube.base.updater.UpRLArgs>`
     - .. autodoc2-docstring:: deepxube.base.updater.UpRLArgs
          :summary:
   * - :py:obj:`UpdateRL <deepxube.base.updater.UpdateRL>`
     -
   * - :py:obj:`UpdateHeur <deepxube.base.updater.UpdateHeur>`
     -
   * - :py:obj:`UpdateHeurV <deepxube.base.updater.UpdateHeurV>`
     -
   * - :py:obj:`UpdateHeurQ <deepxube.base.updater.UpdateHeurQ>`
     -
   * - :py:obj:`UpdatePolicy <deepxube.base.updater.UpdatePolicy>`
     -
   * - :py:obj:`UpdateParser <deepxube.base.updater.UpdateParser>`
     -
   * - :py:obj:`UpdateRLParser <deepxube.base.updater.UpdateRLParser>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_put_from_q <deepxube.base.updater._put_from_q>`
     - .. autodoc2-docstring:: deepxube.base.updater._put_from_q
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InstT <deepxube.base.updater.InstT>`
     - .. autodoc2-docstring:: deepxube.base.updater.InstT
          :summary:
   * - :py:obj:`D <deepxube.base.updater.D>`
     - .. autodoc2-docstring:: deepxube.base.updater.D
          :summary:
   * - :py:obj:`P <deepxube.base.updater.P>`
     - .. autodoc2-docstring:: deepxube.base.updater.P
          :summary:
   * - :py:obj:`UFNsT <deepxube.base.updater.UFNsT>`
     - .. autodoc2-docstring:: deepxube.base.updater.UFNsT
          :summary:
   * - :py:obj:`UFNsHV_T <deepxube.base.updater.UFNsHV_T>`
     - .. autodoc2-docstring:: deepxube.base.updater.UFNsHV_T
          :summary:
   * - :py:obj:`UFNsHQ_T <deepxube.base.updater.UFNsHQ_T>`
     - .. autodoc2-docstring:: deepxube.base.updater.UFNsHQ_T
          :summary:
   * - :py:obj:`UFNsP_T <deepxube.base.updater.UFNsP_T>`
     - .. autodoc2-docstring:: deepxube.base.updater.UFNsP_T
          :summary:
   * - :py:obj:`PS <deepxube.base.updater.PS>`
     - .. autodoc2-docstring:: deepxube.base.updater.PS
          :summary:

API
~~~

.. py:class:: UpArgs
   :canonical: deepxube.base.updater.UpArgs

   .. autodoc2-docstring:: deepxube.base.updater.UpArgs

   .. py:attribute:: procs
      :canonical: deepxube.base.updater.UpArgs.procs
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.procs

   .. py:attribute:: step_max
      :canonical: deepxube.base.updater.UpArgs.step_max
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.step_max

   .. py:attribute:: search_itrs
      :canonical: deepxube.base.updater.UpArgs.search_itrs
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.search_itrs

   .. py:attribute:: up_itrs
      :canonical: deepxube.base.updater.UpArgs.up_itrs
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.up_itrs

   .. py:attribute:: up_gen_itrs
      :canonical: deepxube.base.updater.UpArgs.up_gen_itrs
      :type: typing.Optional[int]
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.up_gen_itrs

   .. py:attribute:: rb
      :canonical: deepxube.base.updater.UpArgs.rb
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.rb

   .. py:attribute:: up_batch_size
      :canonical: deepxube.base.updater.UpArgs.up_batch_size
      :type: typing.Optional[int]
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.up_batch_size

   .. py:attribute:: nnet_batch_size
      :canonical: deepxube.base.updater.UpArgs.nnet_batch_size
      :type: typing.Optional[int]
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.nnet_batch_size

   .. py:attribute:: sync_main
      :canonical: deepxube.base.updater.UpArgs.sync_main
      :type: bool
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.sync_main

   .. py:attribute:: v
      :canonical: deepxube.base.updater.UpArgs.v
      :type: bool
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.v

   .. py:method:: get_up_gen_itrs() -> int
      :canonical: deepxube.base.updater.UpArgs.get_up_gen_itrs

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.get_up_gen_itrs

.. py:function:: _put_from_q(data_l: typing.List[typing.List[numpy.typing.NDArray]], from_q: multiprocessing.Queue, times: deepxube.utils.timing_utils.Times) -> None
   :canonical: deepxube.base.updater._put_from_q

   .. autodoc2-docstring:: deepxube.base.updater._put_from_q

.. py:data:: InstT
   :canonical: deepxube.base.updater.InstT
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.InstT

.. py:data:: D
   :canonical: deepxube.base.updater.D
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.D

.. py:data:: P
   :canonical: deepxube.base.updater.P
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.P

.. py:data:: UFNsT
   :canonical: deepxube.base.updater.UFNsT
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.UFNsT

.. py:data:: UFNsHV_T
   :canonical: deepxube.base.updater.UFNsHV_T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.UFNsHV_T

.. py:data:: UFNsHQ_T
   :canonical: deepxube.base.updater.UFNsHQ_T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.UFNsHQ_T

.. py:data:: UFNsP_T
   :canonical: deepxube.base.updater.UFNsP_T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.UFNsP_T

.. py:class:: Update(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.Update

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsT`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsT`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.updater.D]
      :canonical: deepxube.base.updater.Update.domain_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.domain_type

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfinding.PFNsT]
      :canonical: deepxube.base.updater.Update.pathfind_functions_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.pathfind_functions_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.updater.P]
      :canonical: deepxube.base.updater.Update.pathfind_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.pathfind_type

   .. py:method:: updater_functions_type() -> typing.Type[deepxube.base.updater.UFNsT]
      :canonical: deepxube.base.updater.Update.updater_functions_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.updater_functions_type

   .. py:method:: get_incompat_reason(domain: deepxube.base.domain.Domain, pathfind_fns_t: typing.Type[deepxube.base.pathfind_fns.PFNs], pathfind_t: typing.Type[deepxube.base.pathfinding.PathFind], updater_fns_t: typing.Type[deepxube.base.pathfind_fns.UFNs]) -> typing.Optional[str]
      :canonical: deepxube.base.updater.Update.get_incompat_reason
      :classmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.get_incompat_reason

   .. py:method:: _update_perf(insts: typing.List[deepxube.base.updater.InstT], step_to_pathperf: typing.Dict[int, deepxube.pathfinding.utils.performance.PathFindPerf]) -> None
      :canonical: deepxube.base.updater.Update._update_perf
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._update_perf

   .. py:attribute:: up_args
      :canonical: deepxube.base.updater.Update.up_args
      :type: deepxube.base.updater.UpArgs
      :value: 'UpArgs(...)'

      .. autodoc2-docstring:: deepxube.base.updater.Update.up_args

   .. py:method:: get_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.Update.get_train_shapes_dtypes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.get_train_shapes_dtypes

   .. py:method:: get_train_nnet_par() -> deepxube.base.pathfind_fns.DeepXubeNNetPar
      :canonical: deepxube.base.updater.Update.get_train_nnet_par
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.get_train_nnet_par

   .. py:method:: set_nnet_par_info_l() -> None
      :canonical: deepxube.base.updater.Update.set_nnet_par_info_l

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_nnet_par_info_l

   .. py:method:: set_nnet_par_info(nnet_name: str, nnet_par_info: deepxube.pytorch.nnet_utils.NNetParInfo) -> None
      :canonical: deepxube.base.updater.Update.set_nnet_par_info

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_nnet_par_info

   .. py:method:: start_nnet_runners(device: torch.device, on_gpu: bool) -> None
      :canonical: deepxube.base.updater.Update.start_nnet_runners

      .. autodoc2-docstring:: deepxube.base.updater.Update.start_nnet_runners

   .. py:method:: init_nnet_fns() -> None
      :canonical: deepxube.base.updater.Update.init_nnet_fns

      .. autodoc2-docstring:: deepxube.base.updater.Update.init_nnet_fns

   .. py:method:: clear_nnet_fns() -> None
      :canonical: deepxube.base.updater.Update.clear_nnet_fns

      .. autodoc2-docstring:: deepxube.base.updater.Update.clear_nnet_fns

   .. py:method:: set_main_qs(to_main_q: multiprocessing.Queue, from_main_q: multiprocessing.Queue, q_id: int) -> None
      :canonical: deepxube.base.updater.Update.set_main_qs

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_main_qs

   .. py:method:: start_procs(rb_size: int) -> typing.Tuple[multiprocessing.Queue, typing.List[multiprocessing.Queue]]
      :canonical: deepxube.base.updater.Update.start_procs

      .. autodoc2-docstring:: deepxube.base.updater.Update.start_procs

   .. py:method:: start_update(step_probs: typing.List[int], num_gen: int, train_batch_size: int, device: torch.device, on_gpu: bool) -> None
      :canonical: deepxube.base.updater.Update.start_update

      .. autodoc2-docstring:: deepxube.base.updater.Update.start_update

   .. py:method:: get_update_data(nowait: bool = False) -> typing.List[typing.List[numpy.typing.NDArray]]
      :canonical: deepxube.base.updater.Update.get_update_data

      .. autodoc2-docstring:: deepxube.base.updater.Update.get_update_data

   .. py:method:: end_update() -> typing.Dict[int, deepxube.pathfinding.utils.performance.PathFindPerf]
      :canonical: deepxube.base.updater.Update.end_update

      .. autodoc2-docstring:: deepxube.base.updater.Update.end_update

   .. py:method:: stop_procs() -> None
      :canonical: deepxube.base.updater.Update.stop_procs

      .. autodoc2-docstring:: deepxube.base.updater.Update.stop_procs

   .. py:method:: get_pathfind() -> deepxube.base.updater.P
      :canonical: deepxube.base.updater.Update.get_pathfind

      .. autodoc2-docstring:: deepxube.base.updater.Update.get_pathfind

   .. py:method:: set_targ_update_num(nnet_name: str, targ_update_num: int) -> None
      :canonical: deepxube.base.updater.Update.set_targ_update_num

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_targ_update_num

   .. py:method:: update_runner(to_q: multiprocessing.Queue, from_q: multiprocessing.Queue, rb_size: int) -> None
      :canonical: deepxube.base.updater.Update.update_runner

      .. autodoc2-docstring:: deepxube.base.updater.Update.update_runner

   .. py:method:: _add_instances(pathfind: deepxube.base.updater.P, insts_rem: typing.List[deepxube.base.updater.InstT], batch_size: int, step_probs: typing.List[int], times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.base.updater.Update._add_instances

      .. autodoc2-docstring:: deepxube.base.updater.Update._add_instances

   .. py:method:: _step(pathfind: deepxube.base.updater.P, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.base.updater.Update._step
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._step

   .. py:method:: _step_sync_main(pathfind: deepxube.base.updater.P, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.Update._step_sync_main
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._step_sync_main

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.PFNsT
      :canonical: deepxube.base.updater.Update._get_pathfind_functions
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_pathfind_functions

   .. py:method:: _get_instance_data(instances: typing.List[deepxube.base.updater.InstT], rb_size: int, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.Update._get_instance_data

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_instance_data

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.updater.InstT], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.Update._get_instance_data_norb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.updater.InstT], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.Update._get_instance_data_rb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_instance_data_rb

   .. py:method:: _make_instances(pathfind: deepxube.base.updater.P, steps_gen: typing.List[int], inst_infos: typing.List[typing.Any], times: deepxube.utils.timing_utils.Times) -> typing.List[deepxube.base.updater.InstT]
      :canonical: deepxube.base.updater.Update._make_instances
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._make_instances

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.base.updater.Update._init_replay_buffer
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._init_replay_buffer

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.updater.Update.__repr__

.. py:class:: UpdateHER(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateHER

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.PFNsT`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsT`\ ], :py:obj:`abc.ABC`

   .. py:method:: _step_sync_main(pathfind: deepxube.base.updater.P, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateHER._step_sync_main
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHER._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.updater.InstT], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateHER._get_instance_data_norb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHER._get_instance_data_norb

   .. py:method:: _get_her_goals(instances: typing.List[deepxube.base.updater.InstT], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.updater.InstT], typing.List[deepxube.base.domain.Goal]]
      :canonical: deepxube.base.updater.UpdateHER._get_her_goals

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHER._get_her_goals

.. py:class:: UpdateHasHeurV(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateHasHeurV

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsT`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_heurv_nnet_par() -> deepxube.base.pathfind_fns.HeurVNNetPar
      :canonical: deepxube.base.updater.UpdateHasHeurV.get_heurv_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeurV.get_heurv_nnet_par

   .. py:method:: get_heurv_fn() -> deepxube.base.pathfind_fns.HeurVFn
      :canonical: deepxube.base.updater.UpdateHasHeurV.get_heurv_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeurV.get_heurv_fn

   .. py:method:: _get_targ_heurv_fn() -> deepxube.base.pathfind_fns.HeurVFn
      :canonical: deepxube.base.updater.UpdateHasHeurV._get_targ_heurv_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeurV._get_targ_heurv_fn

.. py:class:: UpdateHasHeurQ(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateHasHeurQ

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsT`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_heurq_nnet_par() -> deepxube.base.pathfind_fns.HeurQNNetPar
      :canonical: deepxube.base.updater.UpdateHasHeurQ.get_heurq_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeurQ.get_heurq_nnet_par

   .. py:method:: get_heurq_fn() -> deepxube.base.pathfind_fns.HeurQFn
      :canonical: deepxube.base.updater.UpdateHasHeurQ.get_heurq_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeurQ.get_heurq_fn

   .. py:method:: _get_targ_heurq_fn() -> deepxube.base.pathfind_fns.HeurQFn
      :canonical: deepxube.base.updater.UpdateHasHeurQ._get_targ_heurq_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeurQ._get_targ_heurq_fn

.. py:class:: UpdateHasPolicy(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateHasPolicy

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_policy_nnet_par() -> deepxube.base.pathfind_fns.PolicyNNetPar
      :canonical: deepxube.base.updater.UpdateHasPolicy.get_policy_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.get_policy_nnet_par

   .. py:method:: get_policy_fn() -> deepxube.base.pathfind_fns.PolicyFn
      :canonical: deepxube.base.updater.UpdateHasPolicy.get_policy_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.get_policy_fn

   .. py:method:: _get_targ_policy_fn() -> deepxube.base.pathfind_fns.PolicyFn
      :canonical: deepxube.base.updater.UpdateHasPolicy._get_targ_policy_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy._get_targ_policy_fn

.. py:data:: PS
   :canonical: deepxube.base.updater.PS
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.PS

.. py:class:: UpdateSup(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateSup

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfind_fns.PFNs`\ , :py:obj:`deepxube.base.updater.PS`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsT`\ ], :py:obj:`abc.ABC`

   .. py:method:: pathfind_functions_type() -> typing.Type[deepxube.base.pathfind_fns.PFNs]
      :canonical: deepxube.base.updater.UpdateSup.pathfind_functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup.pathfind_functions_type

   .. py:method:: _step(pathfind: deepxube.base.updater.PS, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.base.updater.UpdateSup._step

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._step

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfind_fns.PFNs
      :canonical: deepxube.base.updater.UpdateSup._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._get_pathfind_functions

   .. py:method:: _make_instances(pathfind: deepxube.base.updater.PS, steps_gen: typing.List[int], inst_infos: typing.List[typing.Any], times: deepxube.utils.timing_utils.Times) -> typing.List[deepxube.base.updater.InstT]
      :canonical: deepxube.base.updater.UpdateSup._make_instances

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._make_instances

   .. py:method:: _step_sync_main(pathfind: deepxube.base.updater.PS, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateSup._step_sync_main
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._step_sync_main

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.updater.InstT], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateSup._get_instance_data_rb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._get_instance_data_rb

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.base.updater.UpdateSup._init_replay_buffer

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._init_replay_buffer

.. py:class:: UpRLArgs
   :canonical: deepxube.base.updater.UpRLArgs

   .. autodoc2-docstring:: deepxube.base.updater.UpRLArgs

   .. py:attribute:: ub_heur_solns
      :canonical: deepxube.base.updater.UpRLArgs.ub_heur_solns
      :type: bool
      :value: False

      .. autodoc2-docstring:: deepxube.base.updater.UpRLArgs.ub_heur_solns

   .. py:attribute:: lhbl
      :canonical: deepxube.base.updater.UpRLArgs.lhbl
      :type: bool
      :value: False

      .. autodoc2-docstring:: deepxube.base.updater.UpRLArgs.lhbl

.. py:class:: UpdateRL(*args: typing.Any, ub_heur_solns: bool = False, lhbl: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateRL

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsT`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsT`\ ], :py:obj:`abc.ABC`

   .. py:method:: _make_instances(pathfind: deepxube.base.updater.P, steps_gen: typing.List[int], inst_infos: typing.List[typing.Any], times: deepxube.utils.timing_utils.Times) -> typing.List[deepxube.base.updater.InstT]
      :canonical: deepxube.base.updater.UpdateRL._make_instances

      .. autodoc2-docstring:: deepxube.base.updater.UpdateRL._make_instances

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.updater.UpdateRL.__repr__

.. py:class:: UpdateHeur(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateHeur

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsT`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsT`\ ], :py:obj:`abc.ABC`

.. py:class:: UpdateHeurV(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateHeurV

   Bases: :py:obj:`deepxube.base.updater.UpdateHeur`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurV`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHV_T`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsHV_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.UpdateHeurV.get_train_shapes_dtypes

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurV.get_train_shapes_dtypes

   .. py:method:: get_train_nnet_par() -> deepxube.base.pathfind_fns.DeepXubeNNetPar
      :canonical: deepxube.base.updater.UpdateHeurV.get_train_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurV.get_train_nnet_par

   .. py:method:: get_heurv_fn() -> deepxube.base.pathfind_fns.HeurVFn
      :canonical: deepxube.base.updater.UpdateHeurV.get_heurv_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurV.get_heurv_fn

.. py:class:: UpdateHeurQ(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdateHeurQ

   Bases: :py:obj:`deepxube.base.updater.UpdateHeur`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`deepxube.base.updater.UpdateHasHeurQ`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsHQ_T`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsHQ_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.UpdateHeurQ.get_train_shapes_dtypes

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurQ.get_train_shapes_dtypes

   .. py:method:: get_train_nnet_par() -> deepxube.base.pathfind_fns.DeepXubeNNetPar
      :canonical: deepxube.base.updater.UpdateHeurQ.get_train_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurQ.get_train_nnet_par

   .. py:method:: get_heurq_fn() -> deepxube.base.pathfind_fns.HeurQFn
      :canonical: deepxube.base.updater.UpdateHeurQ.get_heurq_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurQ.get_heurq_fn

.. py:class:: UpdatePolicy(domain: deepxube.base.updater.D, pathfind_name_args: str, up_fns: deepxube.base.updater.UFNsT, procs: int = 1, step_max: int = 100, search_itrs: int = 1, up_itrs: int = 100, up_gen_itrs: typing.Optional[int] = None, rb: int = 0, up_batch_size: typing.Optional[int] = None, nnet_batch_size: typing.Optional[int] = None, sync_main: bool = False, v: bool = False, **kwargs: typing.Any)
   :canonical: deepxube.base.updater.UpdatePolicy

   Bases: :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.PFNsP_T`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.InstT`\ , :py:obj:`deepxube.base.updater.UFNsP_T`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_train_nnet_par() -> deepxube.base.pathfind_fns.DeepXubeNNetPar
      :canonical: deepxube.base.updater.UpdatePolicy.get_train_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdatePolicy.get_train_nnet_par

   .. py:method:: get_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.UpdatePolicy.get_train_shapes_dtypes

      .. autodoc2-docstring:: deepxube.base.updater.UpdatePolicy.get_train_shapes_dtypes

   .. py:method:: get_policy_fn() -> deepxube.base.pathfind_fns.PolicyFn
      :canonical: deepxube.base.updater.UpdatePolicy.get_policy_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdatePolicy.get_policy_fn

.. py:class:: UpdateParser()
   :canonical: deepxube.base.updater.UpdateParser

   Bases: :py:obj:`deepxube.base.factory.DelimParser`

   .. py:property:: delim
      :canonical: deepxube.base.updater.UpdateParser.delim
      :type: str

      .. autodoc2-docstring:: deepxube.base.updater.UpdateParser.delim

.. py:class:: UpdateRLParser()
   :canonical: deepxube.base.updater.UpdateRLParser

   Bases: :py:obj:`deepxube.base.updater.UpdateParser`
