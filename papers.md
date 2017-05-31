# PAPERS

## MAIN PAPERS

### A Low-Complexity Detection Scheme for Generalized Spatial Modulation Aided Single Carrier Systems
    - GSM combines SM and V-BLAST
    - compared to PIC-R-SIC, scheme offers near ML detection performance while avoiding complicated matrix operations
    - performance improvement in particular in rank-deficient channel scenarios
    - GSM higher spectral efficiency compared to conventional SM
    - BER performance of GSM dominated by the detectors
    - ZP-aided SM-SC scheme is prefered over CP-aided SM-SC and SM-OFDM schemes in terms of BER
    - PIC-R-SIC: projection matrix, orthogonal to sub-channel matrix
      - GSM sytem: N_u out of N_t tas activated each time slot
      - PIC-R-SIC: if (K+P-1)N_r =< (K-1)N_t, G_I_k may become rank deficient causing an inaccurate detection
      - conventional search algorithms, sphere decoding and M-algorithm aided QR-decompositionnot, not effectively applicable
      - possible rank-deficiency limits application of M-algorithm aided QR-decomposition
      - sphere decoding achieves near-ML performance, but introduces high complexity
      - proposed scheme: QR-decomposition avoided - single stream ML detection employed
      - first step: all possible candidate vectors are generated, based on square Euclidean distances
      - sort in ascended order and M smallest metrics out of Q are selected
      - M legitimate GSM vectors D_1 corresponding to the selected metrics e_1 are obtained
      - second step: second term of receiver signal is detected
      - M^ combinations of x_1 and x_2 corresponding to smallest metrics e_2 are obtained
      - repeat until Kth step
      - complexity of scheme increases with M
      - still achieves complexity reduction of 56% to SD and 78% to PIC-R-SIC and 98% to ML at K=4
      - performance increases with M
      - trade-off between performance and complexity
    - scheme offers better BER performance than PIC-R-SIC with reduced complexity
    - with increasing M, scheme approaches ML detector

## BACKGROUND PAPERS

### M-Algorithm-Based Optimal Detectors for Spatial Modulation (Online / Journal of Communications)
    - M-algorithm reduces complexity of ML detection by combining with the QR decompostition and tree search structure
    - modified M-ML detector proposed with optimal BER performance
    - SM more suitable as an uplink transmission
    - M-algorithm based constellation-reduction algorithm/detector, MCR, proposed
      - since only one ta active, no need for QR decomposition to realize layer search
      - variable complexity depending on the controllable value of M
      - MCR retains M_0 candidate constellation points with smallest metrics to form a new set G
      - then joint detection can be done to achieve optimal performance
      - reduces the complexity when system has at least 2 tas
      - reduction complexity for MCR with ML mainly depends on M_0
      - fixed complexity thanks to M-algorithm, better suited for practical implementation
      - spere decoder (SM-SD) has variable complexity and serial detection structure
      - drawbacks: higher average complexity, complicated metric sorting process
      - complexity of ML detection with high-order modulation increases more significantly than M-ML
      - channel correlation has great negative influence on BER performance, tas difficult to distinguish