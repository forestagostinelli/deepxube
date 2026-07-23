:py:mod:`deepxube.pathfind_fns.fns_core`
========================================

.. py:module:: deepxube.pathfind_fns.fns_core

.. autodoc2-docstring:: deepxube.pathfind_fns.fns_core
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PFNsHeurVC <deepxube.pathfind_fns.fns_core.PFNsHeurVC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurVC
          :summary:
   * - :py:obj:`PFNsHeurQC <deepxube.pathfind_fns.fns_core.PFNsHeurQC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurQC
          :summary:
   * - :py:obj:`PFNsPolicyC <deepxube.pathfind_fns.fns_core.PFNsPolicyC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsPolicyC
          :summary:
   * - :py:obj:`PFNsHeurVPolicyC <deepxube.pathfind_fns.fns_core.PFNsHeurVPolicyC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurVPolicyC
          :summary:
   * - :py:obj:`PFNsHeurQPolicyC <deepxube.pathfind_fns.fns_core.PFNsHeurQPolicyC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurQPolicyC
          :summary:
   * - :py:obj:`HeurVNNetParC <deepxube.pathfind_fns.fns_core.HeurVNNetParC>`
     -
   * - :py:obj:`QOutFixProcessed <deepxube.pathfind_fns.fns_core.QOutFixProcessed>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.QOutFixProcessed
          :summary:
   * - :py:obj:`HeurQNNetParFixOut <deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut>`
     -
   * - :py:obj:`QInProcessed <deepxube.pathfind_fns.fns_core.QInProcessed>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.QInProcessed
          :summary:
   * - :py:obj:`HeurQNNetParIn <deepxube.pathfind_fns.fns_core.HeurQNNetParIn>`
     -
   * - :py:obj:`PolicyNNetParC <deepxube.pathfind_fns.fns_core.PolicyNNetParC>`
     -
   * - :py:obj:`UFNsHeurVC <deepxube.pathfind_fns.fns_core.UFNsHeurVC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UFNsHeurVC
          :summary:
   * - :py:obj:`UFNsHeurQC <deepxube.pathfind_fns.fns_core.UFNsHeurQC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UFNsHeurQC
          :summary:
   * - :py:obj:`UPFNsPolicyC <deepxube.pathfind_fns.fns_core.UPFNsPolicyC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UPFNsPolicyC
          :summary:
   * - :py:obj:`UPFNsHeurVPolicyC <deepxube.pathfind_fns.fns_core.UPFNsHeurVPolicyC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UPFNsHeurVPolicyC
          :summary:
   * - :py:obj:`UPFNsHeurQPolicyC <deepxube.pathfind_fns.fns_core.UPFNsHeurQPolicyC>`
     - .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UPFNsHeurQPolicyC
          :summary:

API
~~~

.. py:class:: PFNsHeurVC
   :canonical: deepxube.pathfind_fns.fns_core.PFNsHeurVC

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNsHeurV`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurVC

.. py:class:: PFNsHeurQC
   :canonical: deepxube.pathfind_fns.fns_core.PFNsHeurQC

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQ`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurQC

.. py:class:: PFNsPolicyC
   :canonical: deepxube.pathfind_fns.fns_core.PFNsPolicyC

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNsPolicy`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsPolicyC

.. py:class:: PFNsHeurVPolicyC
   :canonical: deepxube.pathfind_fns.fns_core.PFNsHeurVPolicyC

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNsHeurVPolicy`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurVPolicyC

.. py:class:: PFNsHeurQPolicyC
   :canonical: deepxube.pathfind_fns.fns_core.PFNsHeurQPolicyC

   Bases: :py:obj:`deepxube.base.pathfind_fns.PFNsHeurQPolicy`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.PFNsHeurQPolicyC

.. py:class:: HeurVNNetParC(domain: deepxube.base.pathfind_fns.D, nnet_input_name: typing.Optional[typing.Tuple[str, str]], nnet_name_args: typing.Optional[str], **kwargs: typing.Any)
   :canonical: deepxube.pathfind_fns.fns_core.HeurVNNetParC

   Bases: :py:obj:`deepxube.base.pathfind_fns.HeurVNNetPar`

.. py:class:: QOutFixProcessed
   :canonical: deepxube.pathfind_fns.fns_core.QOutFixProcessed

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.QOutFixProcessed

   .. py:attribute:: states
      :canonical: deepxube.pathfind_fns.fns_core.QOutFixProcessed.states
      :type: typing.List[deepxube.base.domain.State]
      :value: None

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.QOutFixProcessed.states

.. py:class:: HeurQNNetParFixOut(domain: deepxube.base.pathfind_fns.D, nnet_input_name: typing.Optional[typing.Tuple[str, str]], nnet_name_args: typing.Optional[str], **kwargs: typing.Any)
   :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut

   Bases: :py:obj:`deepxube.base.pathfind_fns.HeurQNNetPar`\ [\ :py:obj:`deepxube.pathfind_fns.fns_core.QOutFixProcessed`\ , :py:obj:`deepxube.base.domain.ActsEnumFixed`\ , :py:obj:`deepxube.base.nnet_input.StateGoalActFixIn`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.ActsEnumFixed]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.domain_type

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.StateGoalActFixIn]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.nnet_input_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.nnet_input_type

   .. py:method:: _check_same_num_acts(actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> None
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut._check_same_num_acts
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut._check_same_num_acts

   .. py:method:: process_inputs(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> deepxube.pytorch.nnet_utils.ProcessedInput[deepxube.pathfind_fns.fns_core.QOutFixProcessed]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.process_inputs

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.process_inputs

   .. py:method:: process_outputs(outs: typing.List[numpy.typing.NDArray], processed: deepxube.pathfind_fns.fns_core.QOutFixProcessed) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.process_outputs

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut.process_outputs

   .. py:method:: _qfix() -> bool
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut._qfix

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut._qfix

   .. py:method:: _out_dim() -> int
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut._out_dim

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParFixOut._out_dim

.. py:class:: QInProcessed
   :canonical: deepxube.pathfind_fns.fns_core.QInProcessed

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.QInProcessed

   .. py:attribute:: states_rep
      :canonical: deepxube.pathfind_fns.fns_core.QInProcessed.states_rep
      :type: typing.List[deepxube.base.domain.State]
      :value: None

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.QInProcessed.states_rep

   .. py:attribute:: split_idxs
      :canonical: deepxube.pathfind_fns.fns_core.QInProcessed.split_idxs
      :type: typing.List[int]
      :value: None

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.QInProcessed.split_idxs

.. py:class:: HeurQNNetParIn(domain: deepxube.base.pathfind_fns.D, nnet_input_name: typing.Optional[typing.Tuple[str, str]], nnet_name_args: typing.Optional[str], **kwargs: typing.Any)
   :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParIn

   Bases: :py:obj:`deepxube.base.pathfind_fns.HeurQNNetPar`\ [\ :py:obj:`deepxube.pathfind_fns.fns_core.QInProcessed`\ , :py:obj:`deepxube.base.domain.Domain`\ , :py:obj:`deepxube.base.nnet_input.StateGoalActIn`\ ]

   .. py:method:: domain_type() -> typing.Type[deepxube.base.domain.Domain]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.domain_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.domain_type

   .. py:method:: nnet_input_type() -> typing.Type[deepxube.base.nnet_input.StateGoalActIn]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.nnet_input_type
      :staticmethod:

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.nnet_input_type

   .. py:method:: process_inputs(states: typing.List[deepxube.base.domain.State], goals: typing.List[deepxube.base.domain.Goal], actions_l: typing.List[typing.List[deepxube.base.domain.Action]]) -> deepxube.pytorch.nnet_utils.ProcessedInput[deepxube.pathfind_fns.fns_core.QInProcessed]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.process_inputs

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.process_inputs

   .. py:method:: process_outputs(outs: typing.List[numpy.typing.NDArray], processed: deepxube.pathfind_fns.fns_core.QInProcessed) -> typing.List[typing.List[float]]
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.process_outputs

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParIn.process_outputs

   .. py:method:: _qfix() -> bool
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParIn._qfix

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParIn._qfix

   .. py:method:: _out_dim() -> int
      :canonical: deepxube.pathfind_fns.fns_core.HeurQNNetParIn._out_dim

      .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.HeurQNNetParIn._out_dim

.. py:class:: PolicyNNetParC(*args: typing.Any, num_samp: int = 0, **kwargs: typing.Any)
   :canonical: deepxube.pathfind_fns.fns_core.PolicyNNetParC

   Bases: :py:obj:`deepxube.base.pathfind_fns.PolicyNNetPar`

.. py:class:: UFNsHeurVC
   :canonical: deepxube.pathfind_fns.fns_core.UFNsHeurVC

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNsHeurV`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UFNsHeurVC

.. py:class:: UFNsHeurQC
   :canonical: deepxube.pathfind_fns.fns_core.UFNsHeurQC

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQ`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UFNsHeurQC

.. py:class:: UPFNsPolicyC
   :canonical: deepxube.pathfind_fns.fns_core.UPFNsPolicyC

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNsPolicy`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UPFNsPolicyC

.. py:class:: UPFNsHeurVPolicyC
   :canonical: deepxube.pathfind_fns.fns_core.UPFNsHeurVPolicyC

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNsHeurVPolicy`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UPFNsHeurVPolicyC

.. py:class:: UPFNsHeurQPolicyC
   :canonical: deepxube.pathfind_fns.fns_core.UPFNsHeurQPolicyC

   Bases: :py:obj:`deepxube.base.pathfind_fns.UFNsHeurQPolicy`

   .. autodoc2-docstring:: deepxube.pathfind_fns.fns_core.UPFNsHeurQPolicyC
