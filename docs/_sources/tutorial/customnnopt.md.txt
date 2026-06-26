(custom_nnopt_section)=
# Creating a Custom Heuristic Function Optimizer and Loss


## Registration, Initialization, and Forward
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
exponential learning rate decay, we implement a step decay.

```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start optim
:end-before: end optim
```


## Custom Loss
```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start loss
:end-before: end loss
```

## Representation
```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start repr
:end-before: end repr
```


## Parser
```{literalinclude} ../../heuristics/resnet_fc_tut.py
:language: python
:class: scroll-code
:start-after: start parser
:end-before: end parser
```

## Training

`deepxube train --domain cube3 --heur resnet_fc_asym.200H_2B_bn_20o --heur_type V --pathfind graph_v --step_max 100 --up_itrs 100 --search_itrs 50 --backup -1 --procs 2 --batch_size 200 --max_itrs 5000 --dir tutorial/heur_asym_loss/models/ --display 50`

`--display 50`: to display the neural network training information every 50 iterations. 

```{literalinclude} ../../tutorial/heur_asym_loss/models/output.txt
:language: none
:class: scroll-code
```

Using `deepxube train_summary --dir tutorial/heur_asym_loss/models/` we can see that the neural network output is below the target 
the majority of the time.

<div style="text-align: center;">
<img src="../_static/gifs/train_summary_heur_asym_loss.gif" alt="Train summary with asymmetric loss" width="100%">
</div>
