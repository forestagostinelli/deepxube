# Known Issues
* Does not handle search case where all reachable states have been explored
* Assumes there is at least one applicable action for each state
* Memory increases even after removing an instance from Search 
until the Search object goes out of scope. Could be due to circular references and the
GC being slow.