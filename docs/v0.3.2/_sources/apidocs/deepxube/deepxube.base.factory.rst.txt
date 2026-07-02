:py:mod:`deepxube.base.factory`
===============================

.. py:module:: deepxube.base.factory

.. autodoc2-docstring:: deepxube.base.factory
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Parser <deepxube.base.factory.Parser>`
     -
   * - :py:obj:`ArgumentSpec <deepxube.base.factory.ArgumentSpec>`
     - .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec
          :summary:
   * - :py:obj:`DelimParser <deepxube.base.factory.DelimParser>`
     -
   * - :py:obj:`Factory <deepxube.base.factory.Factory>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`T <deepxube.base.factory.T>`
     - .. autodoc2-docstring:: deepxube.base.factory.T
          :summary:

API
~~~

.. py:class:: Parser
   :canonical: deepxube.base.factory.Parser

   Bases: :py:obj:`abc.ABC`

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.base.factory.Parser.parse
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.factory.Parser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.base.factory.Parser.help
      :abstractmethod:

      .. autodoc2-docstring:: deepxube.base.factory.Parser.help

.. py:class:: ArgumentSpec
   :canonical: deepxube.base.factory.ArgumentSpec

   .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec

   .. py:attribute:: arg_name_parse
      :canonical: deepxube.base.factory.ArgumentSpec.arg_name_parse
      :type: str
      :value: None

      .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec.arg_name_parse

   .. py:attribute:: arg_name
      :canonical: deepxube.base.factory.ArgumentSpec.arg_name
      :type: str
      :value: None

      .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec.arg_name

   .. py:attribute:: value_type
      :canonical: deepxube.base.factory.ArgumentSpec.value_type
      :type: typing.Optional[typing.Callable[[str], typing.Any]]
      :value: None

      .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec.value_type

   .. py:attribute:: help_msg
      :canonical: deepxube.base.factory.ArgumentSpec.help_msg
      :type: str
      :value: None

      .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec.help_msg

   .. py:attribute:: default
      :canonical: deepxube.base.factory.ArgumentSpec.default
      :type: typing.Optional[typing.Any]
      :value: None

      .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec.default

   .. py:property:: is_boolean
      :canonical: deepxube.base.factory.ArgumentSpec.is_boolean
      :type: bool

      .. autodoc2-docstring:: deepxube.base.factory.ArgumentSpec.is_boolean

.. py:class:: DelimParser()
   :canonical: deepxube.base.factory.DelimParser

   Bases: :py:obj:`deepxube.base.factory.Parser`

   .. py:property:: delim
      :canonical: deepxube.base.factory.DelimParser.delim
      :abstractmethod:
      :type: str

      .. autodoc2-docstring:: deepxube.base.factory.DelimParser.delim

   .. py:method:: add_argument(arg_name_parse: str, arg_name: str, value_type: typing.Optional[typing.Callable[[str], typing.Any]], help_msg: str, default: typing.Optional[typing.Any] = None) -> None
      :canonical: deepxube.base.factory.DelimParser.add_argument

      .. autodoc2-docstring:: deepxube.base.factory.DelimParser.add_argument

   .. py:method:: parse(args_str: str) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.base.factory.DelimParser.parse

      .. autodoc2-docstring:: deepxube.base.factory.DelimParser.parse

   .. py:method:: help() -> str
      :canonical: deepxube.base.factory.DelimParser.help

      .. autodoc2-docstring:: deepxube.base.factory.DelimParser.help

   .. py:method:: _match_arg_name(arg_str_i: str) -> typing.Tuple[str, str]
      :canonical: deepxube.base.factory.DelimParser._match_arg_name

      .. autodoc2-docstring:: deepxube.base.factory.DelimParser._match_arg_name

.. py:data:: T
   :canonical: deepxube.base.factory.T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: deepxube.base.factory.T

.. py:class:: Factory(class_type_str: str)
   :canonical: deepxube.base.factory.Factory

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`deepxube.base.factory.T`\ ]

   .. py:method:: register_class(name: str) -> typing.Callable[[typing.Type[deepxube.base.factory.T]], typing.Type[deepxube.base.factory.T]]
      :canonical: deepxube.base.factory.Factory.register_class

      .. autodoc2-docstring:: deepxube.base.factory.Factory.register_class

   .. py:method:: register_parser(name: str) -> typing.Callable[[typing.Type[deepxube.base.factory.Parser]], typing.Type[deepxube.base.factory.Parser]]
      :canonical: deepxube.base.factory.Factory.register_parser

      .. autodoc2-docstring:: deepxube.base.factory.Factory.register_parser

   .. py:method:: get_parser(name: str) -> typing.Optional[deepxube.base.factory.Parser]
      :canonical: deepxube.base.factory.Factory.get_parser

      .. autodoc2-docstring:: deepxube.base.factory.Factory.get_parser

   .. py:method:: get_kwargs(name: str, args_str: typing.Optional[str]) -> typing.Dict[str, typing.Any]
      :canonical: deepxube.base.factory.Factory.get_kwargs

      .. autodoc2-docstring:: deepxube.base.factory.Factory.get_kwargs

   .. py:method:: get_type(name: str) -> typing.Type[deepxube.base.factory.T]
      :canonical: deepxube.base.factory.Factory.get_type

      .. autodoc2-docstring:: deepxube.base.factory.Factory.get_type

   .. py:method:: build_class(name: str, kwargs: typing.Dict[str, typing.Any]) -> deepxube.base.factory.T
      :canonical: deepxube.base.factory.Factory.build_class

      .. autodoc2-docstring:: deepxube.base.factory.Factory.build_class

   .. py:method:: get_all_class_names() -> typing.List[str]
      :canonical: deepxube.base.factory.Factory.get_all_class_names

      .. autodoc2-docstring:: deepxube.base.factory.Factory.get_all_class_names
