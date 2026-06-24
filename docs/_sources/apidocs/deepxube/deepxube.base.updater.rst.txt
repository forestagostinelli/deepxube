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
   * - :py:obj:`UpdateHasHeur <deepxube.base.updater.UpdateHasHeur>`
     -
   * - :py:obj:`UpdateHasPolicy <deepxube.base.updater.UpdateHasPolicy>`
     -
   * - :py:obj:`UpdateSup <deepxube.base.updater.UpdateSup>`
     -
   * - :py:obj:`UpdateRL <deepxube.base.updater.UpdateRL>`
     -
   * - :py:obj:`UpdateHeur <deepxube.base.updater.UpdateHeur>`
     -
   * - :py:obj:`UpdatePolicy <deepxube.base.updater.UpdatePolicy>`
     -
   * - :py:obj:`UpdateHeurV <deepxube.base.updater.UpdateHeurV>`
     -
   * - :py:obj:`UpdateHeurQ <deepxube.base.updater.UpdateHeurQ>`
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

   * - :py:obj:`FNsH <deepxube.base.updater.FNsH>`
     - .. autodoc2-docstring:: deepxube.base.updater.FNsH
          :summary:
   * - :py:obj:`Inst <deepxube.base.updater.Inst>`
     - .. autodoc2-docstring:: deepxube.base.updater.Inst
          :summary:
   * - :py:obj:`D <deepxube.base.updater.D>`
     - .. autodoc2-docstring:: deepxube.base.updater.D
          :summary:
   * - :py:obj:`P <deepxube.base.updater.P>`
     - .. autodoc2-docstring:: deepxube.base.updater.P
          :summary:
   * - :py:obj:`HNet <deepxube.base.updater.HNet>`
     - .. autodoc2-docstring:: deepxube.base.updater.HNet
          :summary:
   * - :py:obj:`H <deepxube.base.updater.H>`
     - .. autodoc2-docstring:: deepxube.base.updater.H
          :summary:
   * - :py:obj:`PS <deepxube.base.updater.PS>`
     - .. autodoc2-docstring:: deepxube.base.updater.PS
          :summary:

API
~~~

.. py:data:: FNsH
   :canonical: deepxube.base.updater.FNsH
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.FNsH

.. py:class:: UpArgs
   :canonical: deepxube.base.updater.UpArgs

   .. autodoc2-docstring:: deepxube.base.updater.UpArgs

   .. py:attribute:: procs
      :canonical: deepxube.base.updater.UpArgs.procs
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.procs

   .. py:attribute:: up_itrs
      :canonical: deepxube.base.updater.UpArgs.up_itrs
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.up_itrs

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

   .. py:attribute:: ub_heur_solns
      :canonical: deepxube.base.updater.UpArgs.ub_heur_solns
      :type: bool
      :value: False

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.ub_heur_solns

   .. py:attribute:: backup
      :canonical: deepxube.base.updater.UpArgs.backup
      :type: int
      :value: 1

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.backup

   .. py:attribute:: policy_rand_prob
      :canonical: deepxube.base.updater.UpArgs.policy_rand_prob
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.policy_rand_prob

   .. py:attribute:: up_gen_itrs
      :canonical: deepxube.base.updater.UpArgs.up_gen_itrs
      :type: typing.Optional[int]
      :value: None

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.up_gen_itrs

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
      :value: False

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.sync_main

   .. py:attribute:: v
      :canonical: deepxube.base.updater.UpArgs.v
      :type: bool
      :value: False

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.v

   .. py:method:: get_up_gen_itrs() -> int
      :canonical: deepxube.base.updater.UpArgs.get_up_gen_itrs

      .. autodoc2-docstring:: deepxube.base.updater.UpArgs.get_up_gen_itrs

.. py:function:: _put_from_q(data_l: typing.List[typing.List[numpy.typing.NDArray]], from_q: multiprocessing.Queue, times: deepxube.utils.timing_utils.Times) -> None
   :canonical: deepxube.base.updater._put_from_q

   .. autodoc2-docstring:: deepxube.base.updater._put_from_q

.. py:data:: Inst
   :canonical: deepxube.base.updater.Inst
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.Inst

.. py:data:: D
   :canonical: deepxube.base.updater.D
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.D

.. py:data:: P
   :canonical: deepxube.base.updater.P
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.P

.. py:class:: Update(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.Update

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ ], :py:obj:`abc.ABC`

   .. py:method:: domain_type() -> typing.Type[deepxube.base.updater.D]
      :canonical: deepxube.base.updater.Update.domain_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.domain_type

   .. py:method:: functions_type() -> typing.Type[deepxube.base.pathfinding.FNs]
      :canonical: deepxube.base.updater.Update.functions_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.functions_type

   .. py:method:: pathfind_type() -> typing.Type[deepxube.base.updater.P]
      :canonical: deepxube.base.updater.Update.pathfind_type
      :abstractmethod:
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update.pathfind_type

   .. py:method:: _update_perf(insts: typing.List[deepxube.base.updater.Inst], step_to_pathperf: typing.Dict[int, deepxube.pathfinding.utils.performance.PathFindPerf]) -> None
      :canonical: deepxube.base.updater.Update._update_perf
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._update_perf

   .. py:method:: set_nnet_par_info_l_dict() -> None
      :canonical: deepxube.base.updater.Update.set_nnet_par_info_l_dict

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_nnet_par_info_l_dict

   .. py:method:: start_nnet_runners(device: torch.device, on_gpu: bool) -> None
      :canonical: deepxube.base.updater.Update.start_nnet_runners

      .. autodoc2-docstring:: deepxube.base.updater.Update.start_nnet_runners

   .. py:method:: set_nnet_par_info(nnet_name: str, nnet_par_info: deepxube.nnet.nnet_utils.NNetParInfo) -> None
      :canonical: deepxube.base.updater.Update.set_nnet_par_info

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_nnet_par_info

   .. py:method:: clear_nnet_fn_dict() -> None
      :canonical: deepxube.base.updater.Update.clear_nnet_fn_dict

      .. autodoc2-docstring:: deepxube.base.updater.Update.clear_nnet_fn_dict

   .. py:method:: add_nnet_par(nnet_name: str, nnet_par: deepxube.nnet.nnet_utils.NNetPar) -> None
      :canonical: deepxube.base.updater.Update.add_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.Update.add_nnet_par

   .. py:method:: set_nnet_file(nnet_name: str, nnet_file: str) -> None
      :canonical: deepxube.base.updater.Update.set_nnet_file

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_nnet_file

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

   .. py:method:: initialize_fns() -> None
      :canonical: deepxube.base.updater.Update.initialize_fns

      .. autodoc2-docstring:: deepxube.base.updater.Update.initialize_fns

   .. py:method:: get_pathfind() -> deepxube.base.updater.P
      :canonical: deepxube.base.updater.Update.get_pathfind

      .. autodoc2-docstring:: deepxube.base.updater.Update.get_pathfind

   .. py:method:: set_targ_update_num(nnet_name: str, targ_update_num: int) -> None
      :canonical: deepxube.base.updater.Update.set_targ_update_num

      .. autodoc2-docstring:: deepxube.base.updater.Update.set_targ_update_num

   .. py:method:: update_runner(to_q: multiprocessing.Queue, from_q: multiprocessing.Queue, rb_size: int) -> None
      :canonical: deepxube.base.updater.Update.update_runner

      .. autodoc2-docstring:: deepxube.base.updater.Update.update_runner

   .. py:method:: _add_instances(pathfind: deepxube.base.updater.P, insts_rem: typing.List[deepxube.base.updater.Inst], batch_size: int, step_probs: typing.List[int], times: deepxube.utils.timing_utils.Times) -> None
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

   .. py:method:: _get_pathfind_functions() -> deepxube.base.pathfinding.FNs
      :canonical: deepxube.base.updater.Update._get_pathfind_functions
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_pathfind_functions

   .. py:method:: _get_instance_data(instances: typing.List[deepxube.base.updater.Inst], rb_size: int, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.Update._get_instance_data

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_instance_data

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.updater.Inst], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.Update._get_instance_data_norb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_instance_data_norb

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.updater.Inst], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.Update._get_instance_data_rb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._get_instance_data_rb

   .. py:method:: _make_instances(pathfind: deepxube.base.updater.P, steps_gen: typing.List[int], inst_infos: typing.List[typing.Any], times: deepxube.utils.timing_utils.Times) -> typing.List[deepxube.base.updater.Inst]
      :canonical: deepxube.base.updater.Update._make_instances
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._make_instances

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.base.updater.Update._init_replay_buffer
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.Update._init_replay_buffer

   .. py:method:: __repr__() -> str
      :canonical: deepxube.base.updater.Update.__repr__

.. py:class:: UpdateHER(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateHER

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.domain.GoalSampleableFromState`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ ], :py:obj:`abc.ABC`

   .. py:method:: _step_sync_main(pathfind: deepxube.base.updater.P, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateHER._step_sync_main
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHER._step_sync_main

   .. py:method:: _get_instance_data_norb(instances: typing.List[deepxube.base.updater.Inst], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateHER._get_instance_data_norb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHER._get_instance_data_norb

   .. py:method:: _get_her_goals(instances: typing.List[deepxube.base.updater.Inst], times: deepxube.utils.timing_utils.Times) -> typing.Tuple[typing.List[deepxube.base.updater.Inst], typing.List[deepxube.base.domain.Goal]]
      :canonical: deepxube.base.updater.UpdateHER._get_her_goals

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHER._get_her_goals

.. py:data:: HNet
   :canonical: deepxube.base.updater.HNet
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.HNet

.. py:data:: H
   :canonical: deepxube.base.updater.H
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.H

.. py:class:: UpdateHasHeur(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateHasHeur

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.updater.FNsH`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ ], :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.updater.FNsH`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ , :py:obj:`deepxube.base.updater.HNet`\ , :py:obj:`deepxube.base.updater.H`\ ], :py:obj:`abc.ABC`

   .. py:method:: heur_name() -> str
      :canonical: deepxube.base.updater.UpdateHasHeur.heur_name
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeur.heur_name

   .. py:method:: set_heur_nnet(heur_nnet: deepxube.base.updater.HNet) -> None
      :canonical: deepxube.base.updater.UpdateHasHeur.set_heur_nnet

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeur.set_heur_nnet

   .. py:method:: set_heur_file(heur_file: str) -> None
      :canonical: deepxube.base.updater.UpdateHasHeur.set_heur_file

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeur.set_heur_file

   .. py:method:: get_heur_nnet_par() -> deepxube.base.updater.HNet
      :canonical: deepxube.base.updater.UpdateHasHeur.get_heur_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeur.get_heur_nnet_par

   .. py:method:: get_heur_fn() -> deepxube.base.updater.H
      :canonical: deepxube.base.updater.UpdateHasHeur.get_heur_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeur.get_heur_fn

   .. py:method:: _get_heur_fn_from_dict() -> deepxube.base.updater.H
      :canonical: deepxube.base.updater.UpdateHasHeur._get_heur_fn_from_dict

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasHeur._get_heur_fn_from_dict

.. py:class:: UpdateHasPolicy(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateHasPolicy

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ ], :py:obj:`abc.ABC`

   .. py:method:: policy_name() -> str
      :canonical: deepxube.base.updater.UpdateHasPolicy.policy_name
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.policy_name

   .. py:method:: set_policy_samp(policy_samp: int) -> None
      :canonical: deepxube.base.updater.UpdateHasPolicy.set_policy_samp

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.set_policy_samp

   .. py:method:: set_policy_nnet(policy_nnet: deepxube.base.heuristic.PolicyNNetPar) -> None
      :canonical: deepxube.base.updater.UpdateHasPolicy.set_policy_nnet

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.set_policy_nnet

   .. py:method:: set_policy_file(policy_file: str) -> None
      :canonical: deepxube.base.updater.UpdateHasPolicy.set_policy_file

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.set_policy_file

   .. py:method:: get_policy_nnet_par() -> deepxube.base.heuristic.PolicyNNetPar
      :canonical: deepxube.base.updater.UpdateHasPolicy.get_policy_nnet_par

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.get_policy_nnet_par

   .. py:method:: get_policy_fn() -> deepxube.base.heuristic.PolicyFn
      :canonical: deepxube.base.updater.UpdateHasPolicy.get_policy_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHasPolicy.get_policy_fn

.. py:data:: PS
   :canonical: deepxube.base.updater.PS
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.updater.PS

.. py:class:: UpdateSup(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateSup

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`typing.Any`\ , :py:obj:`deepxube.base.updater.PS`\ , :py:obj:`deepxube.base.updater.Inst`\ ], :py:obj:`abc.ABC`

   .. py:method:: functions_type() -> typing.Type[typing.Any]
      :canonical: deepxube.base.updater.UpdateSup.functions_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup.functions_type

   .. py:method:: _step(pathfind: deepxube.base.updater.PS, times: deepxube.utils.timing_utils.Times) -> None
      :canonical: deepxube.base.updater.UpdateSup._step

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._step

   .. py:method:: _get_pathfind_functions() -> typing.Any
      :canonical: deepxube.base.updater.UpdateSup._get_pathfind_functions

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._get_pathfind_functions

   .. py:method:: _make_instances(pathfind: deepxube.base.updater.PS, steps_gen: typing.List[int], inst_infos: typing.List[typing.Any], times: deepxube.utils.timing_utils.Times) -> typing.List[deepxube.base.updater.Inst]
      :canonical: deepxube.base.updater.UpdateSup._make_instances

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._make_instances

   .. py:method:: _step_sync_main(pathfind: deepxube.base.updater.PS, times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateSup._step_sync_main
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._step_sync_main

   .. py:method:: _get_instance_data_rb(instances: typing.List[deepxube.base.updater.Inst], times: deepxube.utils.timing_utils.Times) -> typing.List[numpy.typing.NDArray]
      :canonical: deepxube.base.updater.UpdateSup._get_instance_data_rb
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._get_instance_data_rb

   .. py:method:: _init_replay_buffer(max_size: int) -> None
      :canonical: deepxube.base.updater.UpdateSup._init_replay_buffer

      .. autodoc2-docstring:: deepxube.base.updater.UpdateSup._init_replay_buffer

.. py:class:: UpdateRL(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateRL

   Bases: :py:obj:`deepxube.base.updater.Update`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNs`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ ], :py:obj:`abc.ABC`

   .. py:method:: _make_instances(pathfind: deepxube.base.updater.P, steps_gen: typing.List[int], inst_infos: typing.List[typing.Any], times: deepxube.utils.timing_utils.Times) -> typing.List[deepxube.base.updater.Inst]
      :canonical: deepxube.base.updater.UpdateRL._make_instances

      .. autodoc2-docstring:: deepxube.base.updater.UpdateRL._make_instances

.. py:class:: UpdateHeur(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateHeur

   Bases: :py:obj:`deepxube.base.updater.UpdateHasHeur`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.updater.FNsH`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ , :py:obj:`deepxube.base.updater.HNet`\ , :py:obj:`deepxube.base.updater.H`\ ]

   .. py:method:: get_heur_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.UpdateHeur.get_heur_train_shapes_dtypes
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeur.get_heur_train_shapes_dtypes

   .. py:method:: get_heur_fn() -> deepxube.base.updater.H
      :canonical: deepxube.base.updater.UpdateHeur.get_heur_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeur.get_heur_fn

   .. py:method:: _get_targ_heur_fn() -> deepxube.base.updater.H
      :canonical: deepxube.base.updater.UpdateHeur._get_targ_heur_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeur._get_targ_heur_fn

.. py:class:: UpdatePolicy(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdatePolicy

   Bases: :py:obj:`deepxube.base.updater.UpdateHasPolicy`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsP`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.updater.Inst`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_policy_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.UpdatePolicy.get_policy_train_shapes_dtypes

      .. autodoc2-docstring:: deepxube.base.updater.UpdatePolicy.get_policy_train_shapes_dtypes

   .. py:method:: get_policy_fn() -> deepxube.base.heuristic.PolicyFn
      :canonical: deepxube.base.updater.UpdatePolicy.get_policy_fn

      .. autodoc2-docstring:: deepxube.base.updater.UpdatePolicy.get_policy_fn

.. py:class:: UpdateHeurV(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateHeurV

   Bases: :py:obj:`deepxube.base.updater.UpdateHeur`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHV`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.pathfinding.InstanceNode`\ , :py:obj:`deepxube.base.heuristic.HeurNNetParV`\ , :py:obj:`deepxube.base.heuristic.HeurFnV`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_heur_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.UpdateHeurV.get_heur_train_shapes_dtypes

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurV.get_heur_train_shapes_dtypes

.. py:class:: UpdateHeurQ(domain: deepxube.base.updater.D, pathfind_arg: str, up_args: deepxube.base.updater.UpArgs)
   :canonical: deepxube.base.updater.UpdateHeurQ

   Bases: :py:obj:`deepxube.base.updater.UpdateHeur`\ [\ :py:obj:`deepxube.base.updater.D`\ , :py:obj:`deepxube.base.pathfinding.FNsHQ`\ , :py:obj:`deepxube.base.updater.P`\ , :py:obj:`deepxube.base.pathfinding.InstanceEdge`\ , :py:obj:`deepxube.base.heuristic.HeurNNetParQ`\ , :py:obj:`deepxube.base.heuristic.HeurFnQ`\ ], :py:obj:`abc.ABC`

   .. py:method:: get_heur_train_shapes_dtypes() -> typing.List[typing.Tuple[typing.Tuple[int, ...], numpy.dtype]]
      :canonical: deepxube.base.updater.UpdateHeurQ.get_heur_train_shapes_dtypes

      .. autodoc2-docstring:: deepxube.base.updater.UpdateHeurQ.get_heur_train_shapes_dtypes
