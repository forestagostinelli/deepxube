(custom_nnopt_section)=
# Creating a Custom Heuristic Function Optimizer and Loss

We will create a heuristic function that is optimized via stochastic gradient descent with momentum and that has an asymmetric 
loss that penalizes overestimation more than underestimation.

In the directory in which you run deepxube, create a `heuristics/resnet_fc_tut.py` file.
DeepXube automatically looks in the `heuristics/` folder to see what is registered. This file will be explained part-by-part.


```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
```

```{tip}
For heuristics that are specific to a domain, it is best to define and register them in the domain file, itself. However, for heuristics that
can be re-used across domains, it is best to define and register them in a separate file in the `heuristics/` folder.
```

```{tip}
Since the heuristic is registered, we should be able to see "resnet_fc_asym" with
`deepxube heuristic_info` after it is put in your `heuristics/` folder. 
More specific information can be obtained about the heuristic with 
`deepxube heuristic_info --name resnet_fc_asym`
```


## Registration, Initialization, and Forward

The heuristic function takes one-dimensional data, applies the specified one-hot transformation to it, and feeds this data to a fully-connected 
residual neural network.

```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start registration
:end-before: end forward
```


```{important}
DeepXube expects the first three arguments, `nnet_input: FlatIn, out_dim: int, q_fix: bool` to have these exact names so the 
neural network can be properly initialized.
```

## Custom Optimizer
Instead of using the default Adam optimizer, we implement stochastic gradient descent with momentum. Instead of using the default 
exponential learning rate decay, we implement a step decay. This is done by overriding {mod}`deepxube.base.heuristic.DeepXubeNNet.get_optimizer`
and {mod}`deepxube.base.heuristic.DeepXubeNNet.update_optimizer`.

```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start optim
:end-before: end optim
```


## Custom Loss

By overriding {mod}`deepxube.base.heuristic.HeurNNet.get_loss_and_info`, an asymmetric loss can be implemented. Custom information that 
periodically gets printed to the screen can also be 

```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start loss
:end-before: end loss
```

## Representation

Since the overestimation amount is a parameter given to the class constructor, we can modify the `__repr__` function to also print this 
information.

```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start repr
:end-before: end repr
```

```{tip}
The neural network representation is printed before training to make it easier to determine what neural network architecture was used 
across different runs.
```

## Parser

The parser parses the overestimation penalty along with the dimensionality of the residual network, number of blocks, and whether or not to
perform batch normalization.

```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start parser
:end-before: end parser
```

## Training

`deepxube train --domain cube3 --heur resnet_fc_asym.200H_2B_bn_20o --heur_type V --pathfind graph_v --step_max 100 --up_itrs 100 --search_itrs 50 --backup -1 --procs 2 --batch_size 200 --max_itrs 5000 --dir tutorial/heur_asym_loss/models/ --display 50`

`--display 50`: to display the neural network training information implemented in `get_loss_and_info` every 50 iterations.

```{literalinclude} ../../tutorial/heur_asym_loss/models/output.txt
:language: none
:class: scroll-code
```

Using `deepxube train_summary --dir tutorial/heur_asym_loss/models/` we can see that the neural network output is below the target 
the majority of the time.

<div style="text-align: center;">
<img src="../_static/gifs/train_summary_heur_asym_loss.gif" alt="Train summary with asymmetric loss" width="100%">
</div>
