# DeepXube Basics

## Overview
The objective of DeepXube is to automate the solution of pathfinding problems by using deep reinforcement learning to 
learn heuristic functions that guide heuristic search to solve problems. As a result, user implementation is reduced to the 
definition of the {class}`domain <deepxube.base.domain.Domain>`, {class}`neural network input <deepxube.base.nnet_input.NNetInput>`, 
and {class}`heuristic function <deepxube.base.heuristic.HeurNNet>`. Furthermore, since DeepXube comes with common 
neural networks, such as residual neural networks, the implementation of a heuristic function architecture may not be necessary.

```{tip}
See {ref}`custom_domain_section` for implementing custom domain, neural network input, and heuristic function architecture classes.
```

## Domains

## Training
<div style="text-align: center;">
<img src="../_static/images/deepxube_overview.png" alt="DeepXube Overview" width="100%">
</div>
