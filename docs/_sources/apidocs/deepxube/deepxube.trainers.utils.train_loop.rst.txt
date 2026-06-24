:py:mod:`deepxube.trainers.utils.train_loop`
============================================

.. py:module:: deepxube.trainers.utils.train_loop

.. autodoc2-docstring:: deepxube.trainers.utils.train_loop
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TestArgs <deepxube.trainers.utils.train_loop.TestArgs>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`print_params <deepxube.trainers.utils.train_loop.print_params>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.print_params
          :summary:
   * - :py:obj:`get_curr_itr <deepxube.trainers.utils.train_loop.get_curr_itr>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_curr_itr
          :summary:
   * - :py:obj:`get_curr_update_num <deepxube.trainers.utils.train_loop.get_curr_update_num>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_curr_update_num
          :summary:
   * - :py:obj:`get_pathfind <deepxube.trainers.utils.train_loop.get_pathfind>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_pathfind
          :summary:
   * - :py:obj:`train <deepxube.trainers.utils.train_loop.train>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.train
          :summary:
   * - :py:obj:`get_pathfind_w_instances <deepxube.trainers.utils.train_loop.get_pathfind_w_instances>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_pathfind_w_instances
          :summary:
   * - :py:obj:`test <deepxube.trainers.utils.train_loop.test>`
     - .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.test
          :summary:

API
~~~

.. py:class:: TestArgs
   :canonical: deepxube.trainers.utils.train_loop.TestArgs

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs

   .. py:attribute:: test_states
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.test_states
      :type: typing.List[deepxube.base.domain.State]
      :value: None

      .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs.test_states

   .. py:attribute:: test_goals
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.test_goals
      :type: typing.List[deepxube.base.domain.Goal]
      :value: None

      .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs.test_goals

   .. py:attribute:: search_itrs
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.search_itrs
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs.search_itrs

   .. py:attribute:: pathfinds
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.pathfinds
      :type: typing.List[str]
      :value: None

      .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs.pathfinds

   .. py:attribute:: test_nnet_batch_size
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.test_nnet_batch_size
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs.test_nnet_batch_size

   .. py:attribute:: test_up_freq
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.test_up_freq
      :type: int
      :value: None

      .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs.test_up_freq

   .. py:attribute:: test_init
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.test_init
      :type: bool
      :value: None

      .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.TestArgs.test_init

   .. py:method:: __repr__() -> str
      :canonical: deepxube.trainers.utils.train_loop.TestArgs.__repr__

.. py:function:: print_params(nnet: torch.nn.Module) -> None
   :canonical: deepxube.trainers.utils.train_loop.print_params

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.print_params

.. py:function:: get_curr_itr(train_heur: typing.Optional[deepxube.trainers.train_heur.TrainHeur], train_policy: typing.Optional[deepxube.trainers.train_policy.TrainPolicy]) -> int
   :canonical: deepxube.trainers.utils.train_loop.get_curr_itr

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_curr_itr

.. py:function:: get_curr_update_num(train_heur: typing.Optional[deepxube.trainers.train_heur.TrainHeur], train_policy: typing.Optional[deepxube.trainers.train_policy.TrainPolicy]) -> int
   :canonical: deepxube.trainers.utils.train_loop.get_curr_update_num

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_curr_update_num

.. py:function:: get_pathfind(domain: deepxube.base.domain.Domain, heur_nnet_par: typing.Optional[deepxube.base.heuristic.HeurNNetPar], train_heur: typing.Optional[deepxube.trainers.train_heur.TrainHeur], policy_nnet_par: typing.Optional[deepxube.base.heuristic.PolicyNNetPar], train_policy: typing.Optional[deepxube.trainers.train_policy.TrainPolicy], policy_samp: int, test_nnet_batch_size: int, pathfind_arg: str) -> deepxube.base.pathfinding.PathFind
   :canonical: deepxube.trainers.utils.train_loop.get_pathfind

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_pathfind

.. py:function:: train(domain: deepxube.base.domain.Domain, heur_nnet_par: typing.Optional[deepxube.base.heuristic.HeurNNetPar], update_heur: typing.Optional[deepxube.base.updater.UpdateHeur], policy_nnet_par: typing.Optional[deepxube.base.heuristic.PolicyNNetPar], update_policy: typing.Optional[deepxube.base.updater.UpdatePolicy], policy_samp: int, nnet_dir: str, train_args: deepxube.base.trainer.TrainArgs, test_args: typing.Optional[deepxube.trainers.utils.train_loop.TestArgs] = None, debug: bool = False) -> None
   :canonical: deepxube.trainers.utils.train_loop.train

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.train

.. py:function:: get_pathfind_w_instances(domain: deepxube.base.domain.Domain, heur_nnet_par: typing.Optional[deepxube.base.heuristic.HeurNNetPar], train_heur: typing.Optional[deepxube.trainers.train_heur.TrainHeur], policy_nnet_par: typing.Optional[deepxube.base.heuristic.PolicyNNetPar], train_policy: typing.Optional[deepxube.trainers.train_policy.TrainPolicy], policy_samp: int, test_args: deepxube.trainers.utils.train_loop.TestArgs, pathfind_arg: str) -> deepxube.base.pathfinding.PathFind
   :canonical: deepxube.trainers.utils.train_loop.get_pathfind_w_instances

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.get_pathfind_w_instances

.. py:function:: test(domain: deepxube.base.domain.Domain, heur_nnet_par: typing.Optional[deepxube.base.heuristic.HeurNNetPar], train_heur: typing.Optional[deepxube.trainers.train_heur.TrainHeur], policy_nnet_par: typing.Optional[deepxube.base.heuristic.PolicyNNetPar], train_policy: typing.Optional[deepxube.trainers.train_policy.TrainPolicy], policy_samp: int, test_args: deepxube.trainers.utils.train_loop.TestArgs, writer: torch.utils.tensorboard.SummaryWriter, curr_itr: int) -> None
   :canonical: deepxube.trainers.utils.train_loop.test

   .. autodoc2-docstring:: deepxube.trainers.utils.train_loop.test
