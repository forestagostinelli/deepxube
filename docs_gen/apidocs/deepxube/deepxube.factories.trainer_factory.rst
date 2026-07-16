:py:mod:`deepxube.factories.trainer_factory`
============================================

.. py:module:: deepxube.factories.trainer_factory

.. autodoc2-docstring:: deepxube.factories.trainer_factory
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`get_trainer_from_args <deepxube.factories.trainer_factory.get_trainer_from_args>`
     - .. autodoc2-docstring:: deepxube.factories.trainer_factory.get_trainer_from_args
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`trainer_factory <deepxube.factories.trainer_factory.trainer_factory>`
     - .. autodoc2-docstring:: deepxube.factories.trainer_factory.trainer_factory
          :summary:

API
~~~

.. py:data:: trainer_factory
   :canonical: deepxube.factories.trainer_factory.trainer_factory
   :type: deepxube.base.factory.Factory[deepxube.base.trainer.Train]
   :value: '(...)'

   .. autodoc2-docstring:: deepxube.factories.trainer_factory.trainer_factory

.. py:function:: get_trainer_from_args(nnet_dir: str, updater: deepxube.base.updater.Update, device: torch.device, on_gpu: bool, trainer_name_args: str) -> typing.Tuple[deepxube.base.trainer.Train, str]
   :canonical: deepxube.factories.trainer_factory.get_trainer_from_args

   .. autodoc2-docstring:: deepxube.factories.trainer_factory.get_trainer_from_args
