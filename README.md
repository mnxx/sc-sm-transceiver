# Development and Analysis of a Broadband Single-Carrier Spatial Modulation Transceiver


## Overview

The objective of this work is to implement the Single-Carrier Spatial Modulation scheme and analyze its performance under different conditions. Since the performance of the scheme predominantly depends on the choice of the detector, an integral part of the simulation results are an analysis of the detector design. 

For a more conclusive analysis of the overall scheme and its detector, different channel models have been used. Channel models defined in the LTE standard, which are based on real-world channels, illustrate the viability of the scheme in a realistic context. Moreover, demanding scenarios concerning the antenna setup and other degrees of freedom are simulated and analyzed. Furthermore, the robustness of the scheme to different interference factors, such as frequency-, frame- and phase-offsets, is investigated. In this context, the correct channel estimation and synchronization become an important aspect of the system design. Depending heavily on the diversity of the channel, index modulation schemes rely on a precise channel estimation. While SC schemes are more robust to phase noise than their multi-carrier counterparts, the synchronization process is still essential. Considering the significant number of TAs in LS-MIMO systems, a linear increase of the training overhead is to be expected for classic channel estimation schemes. In order to minimize the overhead, a new approach for a joint channel estimation and synchronization is proposed.

## Structure

### Modules
- 'channel.py' - implementation of different MIMO channel scenarios.
- 'modulation.py' - implementation of linear modulation schemes.
- 'sc_sm_transceiver.py' - implementation of the transceiver, detector and channel estimation.

### Complete Tests
- 'test_scheme.py' - test the detector with perfect channel estimation.
- 'test_system.py' - test the complete transmission scheme.
- 'test_unfiltered.py' - test the complete transmission sans filters.

### Folder 'tests'
This folder contains tests for previous steps in the development.
