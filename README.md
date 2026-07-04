# GVM-GPD: Pre-Trade Algorithmic Trading Model Implementation

This repository contains a Python/PyTorch implementation of the research paper:  
**"A Pre-Trade Algorithmic Trading Model under Given Volume Measures and Generic Price Dynamics (GVM-GPD)"**  
by Jackie Jianhong Shen. [Link to the paper](https://arxiv.org/pdf/1309.5046v2)

---

## Overview

This repository implements a pre-trade algorithmic execution model based on a mean-variance optimization framework. The model introduces several enhancements over traditional methods by incorporating the following key ideas:

- **Volume Measures**: The model uses continuous volume measures (analogous to discrete volume profiles in practice) to represent the execution process. This is modeled through the Participation of Volume (PoV) function, which quantifies the fraction of trading volume allocated over time.
- **Generic Price Dynamics (GPD)**: Instead of assuming standard Brownian motion for price dynamics, this model accounts for more general price behaviors, including memory effects, by directly utilizing auto-covariances. This removes the restrictive Markovian assumption typically associated with Brownian motion.
- **Cost Modeling**: The framework introduces a consistent treatment of four impact cost components:
  1. Temporary market impact
  2. Permanent market impact
  3. Opportunity costs
  4. Market risks
- **Optimization Framework**: The final execution strategy is formulated as a constrained quadratic programming problem in infinite-dimensional Hilbert spaces. The model ensures the existence and uniqueness of optimal solutions using the theory of positive compact operators.

This implementation provides numerical examples to demonstrate the model's behavior and its flexibility in various trading scenarios.

---

## Features

- **Mathematically Rigorous Framework**: Implements the optimization problem as outlined in the paper using advanced mathematical constructs.
- **Customizable Price Dynamics**: Includes support for auto-covariance-based price dynamics, extending beyond simple Brownian motion.
- **Participation Capping**: Incorporates linear constraints on the PoV function, allowing realistic trading strategies.
- **Quadratic Programming Solver**: Leverages numerical solvers to compute optimal execution strategies in the Hilbert space formulation.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gvm-gpd.git
   cd gvm-gpd
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Code Structure

```
gvm-gpd/
├── data/
│   └── sample_volume_profiles.csv    # Example volume profiles for testing
├── models/
│   └── hilbert_space_model.py        # Core implementation of the Hilbert space optimization
├── utils/
│   ├── price_dynamics.py             # Generic price dynamics with support for auto-covariances
│   └── cost_functions.py             # Implementation of the four impact cost components
├── examples/
│   ├── example_config.json           # Configuration for example runs
│   └── run_example.py                # Example usage of the model
├── tests/
│   └── test_model.py                 # Unit tests for model functions
├── README.md                         # Documentation
└── requirements.txt                  # Python dependencies
```

---

## Usage

1. **Prepare Input Data**: Provide volume profiles and price dynamics in the `data/` directory. An example file `sample_volume_profiles.csv` is provided.

2. **Define Configurations**: Update `examples/example_config.json` to specify parameters like:
   - Volume profile
   - Price dynamics model
   - Risk aversion coefficient
   - Participation cap

3. **Run the Model**:
   Execute the provided example script to compute an optimal execution strategy:
   ```bash
   python examples/run_example.py
   ```

4. **Visualize Results**: The script outputs an optimal Participation of Volume (PoV) function and other performance metrics. Visualization scripts can be extended to analyze results further.

---

## Core Concepts

### 1. **Participation of Volume (PoV) Function**
The PoV function represents the trader's relative participation rate in the market over time. It is the Radon-Nikodym derivative of the execution measure with respect to the volume measure.

### 2. **Cost Components**
The model considers four cost components:
- **Temporary Market Impact**: Short-term price changes due to immediate execution.
- **Permanent Market Impact**: Long-term price effects caused by the trade.
- **Opportunity Costs**: Costs incurred from not capturing favorable price movements.
- **Market Risks**: Variability in price movements during execution.

### 3. **Optimization in Hilbert Space**
The execution strategy is cast as a quadratic programming problem in a Hilbert space setting, allowing for a mathematically robust solution.

### 4. **Generic Price Dynamics**
The model supports auto-covariance-based price dynamics to capture memory effects, moving beyond the standard Markovian assumption.

---

## Example Output

After running the example script, the output will include:
- The optimal PoV function over time.
- Total execution cost values (e.g., market impact, opportunity cost, risk).
- Visualizations of the execution strategy and cost components.

---

## Contributing

We welcome contributions to improve the model, add features, or fix issues. Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Description of changes"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## Citation

If you use this work in your research, please cite the original paper:

```
@article{shen2013pretrade,
  title={A Pre-Trade Algorithmic Trading Model under Given Volume Measures and Generic Price Dynamics (GVM-GPD)},
  author={Jackie Jianhong Shen},
  journal={arXiv preprint arXiv:1309.5046},
  year={2013}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.