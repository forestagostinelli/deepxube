:py:mod:`deepxube.factories.pathfind_fns_factory`
=================================================

.. py:module:: deepxube.factories.pathfind_fns_factory

.. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_dx_nnet_par <deepxube.factories.pathfind_fns_factory.get_dx_nnet_par>`
     - .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.get_dx_nnet_par
          :summary:
   * - :py:obj:`get_path_up_fns <deepxube.factories.pathfind_fns_factory.get_path_up_fns>`
     - .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.get_path_up_fns
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`deepxube_nnet_par_factory <deepxube.factories.pathfind_fns_factory.deepxube_nnet_par_factory>`
     - .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.deepxube_nnet_par_factory
          :summary:
   * - :py:obj:`pathfind_fns_factory <deepxube.factories.pathfind_fns_factory.pathfind_fns_factory>`
     - .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.pathfind_fns_factory
          :summary:
   * - :py:obj:`updater_fns_factory <deepxube.factories.pathfind_fns_factory.updater_fns_factory>`
     - .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.updater_fns_factory
          :summary:

API
~~~

.. py:data:: deepxube_nnet_par_factory
   :canonical: deepxube.factories.pathfind_fns_factory.deepxube_nnet_par_factory
   :type: deepxube.base.factory.Factory[deepxube.base.pathfind_fns.DeepXubeNNetPar]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.deepxube_nnet_par_factory

.. py:data:: pathfind_fns_factory
   :canonical: deepxube.factories.pathfind_fns_factory.pathfind_fns_factory
   :type: deepxube.base.factory.FactoryAutoBuild[deepxube.base.pathfind_fns.PFNs]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.pathfind_fns_factory

.. py:data:: updater_fns_factory
   :canonical: deepxube.factories.pathfind_fns_factory.updater_fns_factory
   :type: deepxube.base.factory.FactoryAutoBuild[deepxube.base.pathfind_fns.UFNs]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.updater_fns_factory

.. py:function:: get_dx_nnet_par(domain: deepxube.base.domain.Domain, domain_name: str, nnet_par_name_args: str, nnet_name_args: typing.Optional[str]) -> typing.Tuple[deepxube.base.pathfind_fns.DeepXubeNNetPar, str]
   :canonical: deepxube.factories.pathfind_fns_factory.get_dx_nnet_par

   .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.get_dx_nnet_par

.. py:function:: get_path_up_fns(domain: deepxube.base.domain.Domain, domain_name: str, fn_name_args_l: typing.List[str], device: torch.device, nnet_files: typing.Optional[typing.List[typing.Optional[str]]] = None, nnet_batch_size: typing.Optional[int] = None) -> typing.Tuple[deepxube.base.pathfind_fns.PFNs, deepxube.base.pathfind_fns.UFNs]
   :canonical: deepxube.factories.pathfind_fns_factory.get_path_up_fns

   .. autodoc2-docstring:: deepxube.factories.pathfind_fns_factory.get_path_up_fns
