# DVDF
Code for ICML 2026 submission &lt;Efficient Cross-Domain Offline RL with Dynamics- and Value-Aligned Data Filtering>

The implementation is heavily based on  [ODRL repository]( https://github.com/OffDynamicsRL/off-dynamics-rl).

* Prepare your source domain datasets in .hdf5 format in `DVDF/dataset/source/`.
* Prepare your environments as required by ODRL.
* Run the script `run_offline.sh` to pre-train the advantage function.
* Run the srcipt `run_dvdf.sh` to train and evaluate DVDF