# Known Issues
* `deepxube` command may not work on Windows. Can still use `python -m deepxube` to get the same functionality.
* Does not handle search case where all reachable states have been explored
* Assumes there is at least one applicable action for each state. Can be addressed by having an action that does not 
state in cases there are no applicable actions.
* Solver get_num_ground_rules behaves differently if self.ctl_rand.statistics is 
looked at before self.ctl_rand.solve()