# Balancing Training Data Difficulty

Although problem instance generation need not use the specified number of steps when generating problem instances, some implementations that do
can use the number of steps used to generate a instance as an approximation of the difficulty of that problem instance. If the maximum number of
steps is too small, then the heuristic function will not generalize to more difficult problems. If the maximum number of steps is too large,
then the heuristic function may not see enough solved instances during training, resulting in increasingly large cost-to-go targets.

To address this, the maximum number of steps can be initialized to 1 and doubled everytime the average percentage of problems solved reached 50%
until a pre-determined absolute maximum number of steps is reached (see {cite:p}`agostinelli2026data`).

We will experiment with the 6x6 sliding tile puzzle (i.e. the 35-puzzle) when training with a maximum number of steps of 10,000.

## Without Balancing

`deepxube train --domain npuzzle.6 --fn heurv,resnet_fc.200H_2B_bn --pathfind graph --up up_rl.10000sm_100up_100sitrs_lhbl_2p --tr tr_h.200bs_5000maxit --dir tutorial/npuzzle6/10000sm/`

```{literalinclude} ../../tutorial/npuzzle6/10000sm/output.txt
:language: none
:class: scroll-code
```

Note how there is rarely an instance that is solved, the cost-to-go continues to increase, and the percentage solved does not improve.

`deepxube train_summary --dir tutorial/npuzzle6/10000sm/`

```{image} ../../tutorial/npuzzle6/10000sm/train_summary.gif
:alt: Search demo animation no balancing
:width: 600px
:align: center
```

## With Balancing

`deepxube train --domain npuzzle.6 --fn heurv,resnet_fc.200H_2B_bn --pathfind graph --up up_rl.10000sm_100up_100sitrs_lhbl_2p --tr tr_h.200bs_5000maxit_bal --dir tutorial/npuzzle6/10000sm_bal/`

- `_bal` to balance training data

```{literalinclude} ../../tutorial/npuzzle6/10000sm_bal/output.txt
:language: none
:class: scroll-code
```

There are significantly more solved instances seen during training and search success and percentage solved increases over training time.

`deepxube train_summary --dir tutorial/npuzzle6/10000sm_bal/`

```{image} ../../tutorial/npuzzle6/10000sm_bal/train_summary.gif
:alt: Search demo animation with balancing
:width: 600px
:align: center
```
