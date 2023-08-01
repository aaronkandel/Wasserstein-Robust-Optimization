# Wasserstein Robust Optimization
Starter pack for distributionally robust chance-constrained optimization using the Wasserstein distance. This is specifically an unofficial implementation of the equivalent chance-constraint reformulation referenced in the following paper:

C. Duan, W. Fang, L. Jiang, L. Yao and J. Liu, "Distributionally Robust Chance-Constrained Approximate AC-OPF With Wasserstein Metric," in IEEE Transactions on Power Systems, vol. 33, no. 5, pp. 4924-4936, Sept. 2018, doi: 10.1109/TPWRS.2018.2807623.

# Installation

Consider the following steps to install this directory, and get started solving your own distributionally robust optimization programs!

1. clone repository
2. Navigate to the root directory, and run the following commands:
   a. pip install -r requirements.txt <br />
   b. pip install -e .

Now, you are ready to get started running code.

To test the DRO reformulation algorithm, there are several scripts already included in the repo. These include "vectorExample.py" and "scalarExample.py." 
