# PAPERS

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