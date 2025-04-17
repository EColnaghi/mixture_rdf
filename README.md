This repository provides a collection of theoretical functions for computing thermodynamic and structural properties of hard-sphere fluids, and for analyzing and comparing them with the results of MD simulations, with a focus on polydisperse systems of hard-spheres and hard-spheres polymers. The code implements analytical and semi-analytical models based on classical statistical mechanics and integral equation theories.
---

## 📦 Main Features(theoretical)

- **Equations of state** for monodisperse and polydisperse systems:
  - Carnahan-Starling (monodisperse)
  - Boublik-Mansoori-Carnahan-Starling (polydisperse)
  - Ideal gas limit
- **Pair correlation functions**:
  - Monodisperse systems via the Ornstein-Zernike equation with Percus-Yevick closure
  - Bidisperse systems in Laplace space
  - Full RFA (Rational Function Approximation) approach for polydisperse systems
- **Laplace and real-space transforms** of radial distribution functions
- **Calculation of second virial coefficients**
- **Output of effective potentials** in a LAMMPS-compatible format

  ## 📦 Main Features(data analysis)

- Compute **radial distribution functions** (`g(r)`) from trajectory files
- Analyze **gyration radius** and its evolution over time
- Extract **end-to-end distances** and **Kuhn length distributions** for polymers
- Calculate **mean squared displacement** for atoms or chains
- Measure **density fluctuations** and **local densities**
- Generate visualizations and phase diagrams for crowding behavior
---

## Dependencies

- `numpy`
- `scipy`
- `matplotlib`
- `sympy` (for symbolic Laplace RFA, not currently functional)
- `tqdm`
- `re`, `os`, `pdb` (standard Python libraries)

---

## Functions

- **carnahan_starling**	Carnahan-Starling equation of state
- **boublik_mansoori_CS**	Equation of state for polydisperse systems
- **monodisperse_correlation_function**	Pair correlation from OZ+PY (monodisperse)
- **laplace_binary_correlation_function**	Laplace-space g(s) for bidisperse systems
- **alpha_RFA, laplace_RFA, rdf_RFA**	Full RFA-based structural analysis
- **second_virial_coefficient_from_rdf**	Second virial from g(r)
- **write_potential_table**	Export potentials for LAMMPS simulations
- **radial_distribution** **radial_distribution_done_right** Calculates pairwise distances and builds RDF
- **gyration_radius** Computes the radius of gyration over time
- **end_to_end_distance** Computes polymer end-to-end distance per frame 
- **mean_squared_displacement** **mean_squared_displacement_polymer** Tracks atom/chain diffusion 
- **kuhn_length_distribution** | Calculates bond lengths for polymer chains 
- **density_fluctuation** Measures spatial fluctuations in particle density 
- **swelling_parameter_sigma** Computes a measure of polymer swelling over blocks 

---
