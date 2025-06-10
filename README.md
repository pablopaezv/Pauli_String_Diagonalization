Description:
This script implements a tableau-based framework for representing commuting Pauli Hamiltonians. It provides operations
to manipulate those tableaus and includes algorithms to find a diagonalizing circuit for any such Hamiltonian.
The notation and some of the methods/algorithms are based on the paper "Improved simulation of stabilizer circuits"
from Scott Aaronson and Daniel Gottesman (2004), which are further studied in "Circuit optimization of Hamiltonian
simulation by simultaneous diagonalization of Pauli clusters" from Ewout van den Berg and Kristan Temme (2020).

The main idea of the papers named above is to store each Pauli term of a given Hamiltonian as a row in two binary
matrices X and Z (plus a single binary in the sign vector s). Depending on the Hamiltonian, which acts on n qubits
and has m terms, the matrices X and Z will have shape (m, n). To diagonalize the Hamiltonian, the algorithms determine
a set of operations that can be applied to the tableau, such that it clears the X block, where each operation 
corresponds to a physical Clifford gate (Hadamard, Phase, CNOT, CZ) or a virtual operation (swaps of rows and columns 
and sweeps of rows). Once the intial tableau is diagonalized (i.e. the X block is cleared), the program extracts only 
the physical Clifford gates from the set of operations and applies them to a fresh tableau to find the actual form of 
the final diagonalized Hamiltonian.

In this script, we provide functions to automatically build the tableaus for various quantum error-correcting models
(2D Toric code, Color code on honeycomb lattice, Haah’s code, 3D Toric code, X-cube model, Rotated surface code and
Triangle and Cube subsystem toric codes) of variable lattice size and to run the diagonalization algorithms on them.
It also provides a “checker” mode that tests whether the final Z-only tableaus match the expected canonical forms 
which are claimed in our paper for the different models.


The following code is structured in several sections:

----------------
1. Definition Tabelau Class

Provides a data structure for storing a list of Pauli terms (in X/Z binary form plus a sign vector)
and implements:

- Virtual tableau operations (row/column swaps, row sweeps) used by the diagonalization routines
    
- Physical Clifford gate operations (Hadamard, Phase, CNOT, CZ) also used by the diagonalization routines
      (Both physical and virtual gates update the tableau and are logged in gate logs)
      
- Two core diagonalization algorithms (clearing the X block of the tableau) to diagonalize any commuting 
      Pauli Hamiltonian from the paper [van den Berg and Temme, 2020]
      
- A method to extract and apply from a gate log only the physical Clifford gates on a fresh tableau.
    
- A “simplify_Z” procedure similar to gaussian elimination that further simplifies the tableau (where the X block 
      is already cleared) into a canonical form.
      
- Utilities for printing and for checking whether a fully simplified Z tableau matches the expected form 
      (proposed in the accompanying paper) for a variety of error‐correcting code models (Toric code, Haah’s code, etc.).

----------------
2. Helper Functions

Contains miscellaneous routines that support building and validating tableaus, including:
- A logging function to write progress and error messages both to the console and to an output file.

- Indexing functions that map 2D/3D lattice coordinates under various boundary conditions into
      single‐integer spin indeces, e.g. “onedposhaah”, “onedpostoric”, “onedposXcube”, “onedpossurface”.
      (These coordinate mappers are used by the model‐specific builder functions from section 3 
      to place each Pauli operator at the correct qubit index when constructing the X and Z binary matrices 
      for a given model).

----------------
3. Model-Specific Tableau Builders

Defines builder functions that return all building block of a tableau, i.e. (X, Z, s) for a given lattice size L
Each interaction model has its own tableau builder function
- toric_code_matrices: 2D Toric code on a 2D L×L square lattice with periodic boundary conditions.
- color_honeycomb_matrices: Color code on a 2D honeycomb lattice with periodic boundary conditions.
- toric_3D_matrices: 3D Toric code on a 3D cubic square lattice with periodic boundary conditions.
- haah_matrices: Haah’s code on a 3D cubic lattice with periodic boundary conditions, but here with two qubits 
      at every vertex.
- X_cube_matrices: X‐cube model on a 3D lattice with cylindrical boundary conditions.
- triangle_subsystem_toric_matrices & cube_subsystem_toric_matrices: Triangle and cube subsystem toric codes
        on 3D cubic lattices with periodic boundary conditions.
- rotated_surface_matrices: Rotated surface code on a 2D square lattice with open boundary conditions.

Each builder initializes binary matrices X and Z (and a sign vector s of zeros) and then populates them
row by row according to the geometry and coloring of the operators acting on the lattice.

----------------
4. Main Routines

Provides a command-line interface for two modes:
- single run: Instantiates the appropriate builder at its default “single_size” L, prints the initial tableau,
        runs the two‐step diagonalization, applies only the physical Clifford gates from this process 
        to a new initial tableau, applies the Z‐simplification onto the "physical" tableau and prints 
        the final tableau and a boolean check for consistency with the predicted form.
- checker: Iterates over a range of lattice sizes (and depending on the command also for different models),
        runs full diagonalization and simplification for each size, and logs any failures (when the final Z
        tableau does not match the expected form) both to the console and to “output.txt”.
The “model_config” dictionary centralizes:
- Which builder function to call for each interaction model (e.g. toric_code_matrices, haah_matrices, ...).
- Which lattice sizes to use for a single run and which ranges for the checker.
- Any size constraints

The “main()” function parses sys.argv to dispatch either “single_run” or “checker” as requested.

----------------
Usage:
1. To perform a single diagonalization at the default size (single_size is set in model_config) for a given model:

    ```python path/to/program_name.py single_run <model_name>```

    Example:

    ```python program.py single_run toric_code```

3. To run the checker over one specific model, verifying that over a range of lattice sizes (the range is set
       via checker_sizes in model_config), the digonalization algorithm leads to the predicted final tableau:

    ```python path/to/program_name.py checker <model_name>```

   Example:

   ```python program.py checker haahs_code```

5. To run the checker on every model in `model_config` over a range of lattice sizes:

   ```python path/to/program.py checker```

7. Output:
- 'single_run' prints initial/final tableaus and whether the final form is correct.
- 'checker' logs progress and any “ERROR” lines into `output.txt` (and also prints the progess to the console).
