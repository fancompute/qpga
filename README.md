# Quantum programmable gate arrays

![](/assets/qpga_design.png)

## Overview

This repository contains the gradient-based optimization code for the paper "[Universal programmable photonic architecture for quantum information processing](https://link.aps.org/doi/10.1103/PhysRevA.101.042319)". In this paper, we present a photonic integrated circuit architecture for a quantum programmable gate array (QPGA) capable of preparing arbitrary quantum states and operators. The architecture consists of a lattice of phase-modulated Mach-Zehnder interferometers, which perform rotations on path-encoded photonic qubits, and embedded quantum emitters, which use a two-photon scattering process to implement a deterministic controlled-Z operation between adjacent qubits. By appropriately setting phase shifts within the lattice, the device can be programmed to implement any quantum circuit without hardware modifications. We provide algorithms for exactly preparing arbitrary quantum states and operators on the device and we show that gradient-based optimization can train a simulated QPGA to automatically implement highly compact approximations to important quantum circuits with near-unity fidelity.


## Dependencies

- `Python >3.6`
- `TensorFlow 1.14`
- `SQUANCH >1.1`
- `numpy`
- `scipy`
- `matplotlib`


## Components

The structure of the repository is as follows:

- `qpga_figures.ipynb`: a notebook containing the code to generate the figures used in the paper
- `qpga`
    - `callbacks.py`: provides Keras-style callbacks for recording the logical operators and states implemented by the simulated QPGA over the course of training
    - `circuits.py`: collection of quantum circuits simulated in `SQUANCH` used in preparing training data
    - `constants.py`: various constants used in the repository
    - `fidelity_search.py`: trains QPGAs of increasing depth to match input to output data to a desired fidelity
    - `linalg.py`: linear algebraic helper functions
    - `model.py`: contains the main TensorFlow model for simulating a QGPA
    - `plotting.py`: helper functions for generating figures used in the paper
    - `state_preparation.py`: helper functions for preparing quantum state vectors
    - `training.py`: contains a helper function for instantiating and compiling a QGPA to fit to input/output data
    - `utils.py`: miscellaneous utilities 
- `scripts`
    - `run_ghz_state_fidelity.py`: script to find a QPGA which prepares GHZ states to a desired fidelity
    - `run_qft_fidelity_search.py`: script to find a QPGA which implements a quantum Fourier transform to within a desired fidelity

Training histories are written to `h5py` files which are not included in this repository, but are available upon request from the first author.


## Supplementary materials

Supplementary materials for the arXiv version of the paper are listed below. Click an image to view it in higher resolution, or click the [source] link to download the original file.

### S1: Conceptual animation of the two-photon scattering process described in Section IIB

[![](https://thumbs.gfycat.com/BlondDefiniteGoitered-size_restricted.gif)](https://gfycat.com/blonddefinitegoitered)

[[source]](https://github.com/fancompute/qpga/raw/master/assets/gate_animation.mp4)

This animation depicts the four steps of the two-photon scattering process:
1. Photon $A$ at frequency $\omega$ causes the atom, which is initialized in state $\ket{1}$, to partially transition from $\ket{1} \rightarrow \ket{3}$ with an amplitude of $\ket{3}$ corresponding to the photon occupancy in the waveguide. This emits an auxiliary photon $A'$ with frequency $\omega'$, which is reflected by one of the narrow-band mirrors and travels down the delay line. 
2. While photon $A'$ is in the delay line, photon $B$, also at frequency $\omega$, is injected into the system. Interaction with the $\ket{1}$ component of the atomic states results in the transition $\ket{1}\rightarrow\ket{3}$ and releases an auxiliary photon $B'$ with frequency $\omega'$ down the delay line, while interaction with the $\ket{3}$ component imparts a $\pi$ phase shift onto $B$ and reflects it back into the waveguide.
3. Photon $A'$ arrives back at the 4LS after traversing the delay line. By time reversal arguments, sending the output photon $A'$ back into the atom retrieves photon $A$, which exits the inner cell through its original waveguide.
4. Photon $B'$ arrives back at the 4LS, retrieving photon $B$ as in step 3.

### S2: Animation depicting optimization of a QPGA to perform a five-qubit quantum Fourier transform

[![](https://thumbs.gfycat.com/SafeDigitalAbyssiniangroundhornbill-size_restricted.gif)](https://gfycat.com/safedigitalabyssiniangroundhornbill)

[[source]](https://github.com/fancompute/qpga/raw/master/assets/qft_training.mp4)

Optimization of a QPGA to prepare a quantum Fourier transform on five input qubits. (Top left) The operator implemented by the QPGA at each point in training. The square array represents the magnitude (relative to the maximum element) and phase of the projection of the operator onto the lexicographically-ordered computational basis states, encoded in the respective size and hue of the squares. (Top right) The target 5-qubit QFT operator. (Bottom) Fidelity between the implemented and target operator over the course of training. 


## Citing

If you found this paper or repository useful, please cite us using:

```
@article{Bartlett2020Universal,
  title = {Universal programmable photonic architecture for quantum information processing},
  author = {Bartlett, Ben and Fan, Shanhui},
  journal = {Phys. Rev. A},
  volume = {101},
  issue = {4},
  pages = {042319},
  numpages = {15},
  year = {2020},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevA.101.042319},
  url = {https://link.aps.org/doi/10.1103/PhysRevA.101.042319}
}
```

