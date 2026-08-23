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
   * - :py:obj:`register_nnet_input_parser <deepxube.factories.nnet_input_factory.register_nnet_input_parser>`
     - .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input_parser
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
   * - :py:obj:`get_nnet_input_from_arg <deepxube.factories.nnet_input_factory.get_nnet_input_from_arg>`
     - .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_nnet_input_from_arg
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
   :type: typing.Dict[str, deepxube.base.factory.Factory[deepxube.base.nnet_input.NNetInput]]
   :value: None

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory._nnet_input_registry

.. py:function:: register_nnet_input(domain_name: str, nnet_input_name: str) -> typing.Callable[[typing.Type[deepxube.base.nnet_input.NNetInput]], typing.Type[deepxube.base.nnet_input.NNetInput]]
   :canonical: deepxube.factories.nnet_input_factory.register_nnet_input

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input

.. py:function:: register_nnet_input_parser(domain_name: str, nnet_input_name: str) -> typing.Callable[[typing.Type[deepxube.base.factory.Parser]], typing.Type[deepxube.base.factory.Parser]]
   :canonical: deepxube.factories.nnet_input_factory.register_nnet_input_parser

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input_parser

.. py:function:: get_domain_nnet_input_keys(domain_name: str) -> typing.List[str]
   :canonical: deepxube.factories.nnet_input_factory.get_domain_nnet_input_keys

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_domain_nnet_input_keys

.. py:function:: get_nnet_input_t(domain_name: str, nnet_input_name: str) -> typing.Type[deepxube.base.nnet_input.NNetInput]
   :canonical: deepxube.factories.nnet_input_factory.get_nnet_input_t

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_nnet_input_t

.. py:function:: register_nnet_input_dynamic() -> None
   :canonical: deepxube.factories.nnet_input_factory.register_nnet_input_dynamic

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.register_nnet_input_dynamic

.. py:function:: get_nnet_input_from_arg(domain: deepxube.base.domain.Domain, domain_name: str, nnet_input_name_args: str) -> deepxube.base.nnet_input.NNetInput
   :canonical: deepxube.factories.nnet_input_factory.get_nnet_input_from_arg

   .. autodoc2-docstring:: deepxube.factories.nnet_input_factory.get_nnet_input_from_arg
