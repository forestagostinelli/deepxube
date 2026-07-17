:py:mod:`deepxube.factories.nnet_input_factory`
===============================================

.. py:module:: deepxube.factories.nnet_input_factory

.. autodoc2-docstring:: deepxube.factories.nnet_input_factory
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`register_nnet_input <deepxube.factories.nnet_input_factory.register_nnet_input>`
     - .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input
          :summary:
   * - :py:obj:`get_domain_nnet_input_keys <deepxube.factories.nnet_input_factory.get_domain_nnet_input_keys>`
     - .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_domain_nnet_input_keys
          :summary:
   * - :py:obj:`get_nnet_input_t <deepxube.factories.nnet_input_factory.get_nnet_input_t>`
     - .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_nnet_input_t
          :summary:
   * - :py:obj:`register_nnet_input_dynamic <deepxube.factories.nnet_input_factory.register_nnet_input_dynamic>`
     - .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input_dynamic
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_nnet_input_registry <deepxube.factories.nnet_input_factory._nnet_input_registry>`
     - .. autodoc2-docstring:: deepxube.factories.nnet_input_factory._nnet_input_registry
          :summary:

API
~~~

.. py:data:: _nnet_input_registry
   :canonical: deepxube.factories.nnet_input_factory._nnet_input_registry
   :type: typing.Dict[typing.Tuple[str, str], typing.Type[deepxube.base.nnet_input.NNetInput]]
   :value: None

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory._nnet_input_registry

.. py:function:: register_nnet_input(domain_name: str, nnet_input_name: str) -> typing.Callable[[typing.Type[deepxube.base.nnet_input.NNetInput]], typing.Type[deepxube.base.nnet_input.NNetInput]]
   :canonical: deepxube.factories.nnet_input_factory.register_nnet_input

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input

.. py:function:: get_domain_nnet_input_keys(domain_name: str) -> typing.List[typing.Tuple[str, str]]
   :canonical: deepxube.factories.nnet_input_factory.get_domain_nnet_input_keys

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_domain_nnet_input_keys

.. py:function:: get_nnet_input_t(key: typing.Tuple[str, str]) -> typing.Type[deepxube.base.nnet_input.NNetInput]
   :canonical: deepxube.factories.nnet_input_factory.get_nnet_input_t

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_nnet_input_t

.. py:function:: register_nnet_input_dynamic() -> None
   :canonical: deepxube.factories.nnet_input_factory.register_nnet_input_dynamic

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input_dynamic
