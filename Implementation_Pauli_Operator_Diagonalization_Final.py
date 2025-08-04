import numpy as np
import copy
import os
import math
import sys
print("Current working directory:", os.getcwd())

"""
This code is published as part of the paper:
"Efficient and simple Gibbs sampling of the 2D toric code via duality to classical Ising chains" (arXiv:2508.00126)
Authors: Pablo Páez Velasco, Niclas Schilling, Samuel O. Scalet, Frank Verstraete and Ángela Capel

This code in under MIT License
Copyright (c) 2025 Pablo Páez Velasco, Niclas Schilling, Samuel O. Scalet, Frank Verstraete, and Ángela Capel
Date: June 2025
"""

#--------------------------------
# Defintion of the Tableau Class
#--------------------------------

class Tableau:
    def __init__(self, X, Z, s):
        """
        Initialize a tableau.
        Input:
        X, Z: numpy arrays of shape (m, n) with entries in {0,1}
        s: numpy array of length m (0 for +, 1 for -)
        
        Other parameters, which can be retrieved by self.[parameter]:
        m, n: dimensions of the X and Z matrices (m rows, n columns)
        gate_log, gate_log_column_indexing: list of tuples (gate_name, tableau_column_indices) of gates already applied
        (see comments below for details on the differences between the two gate logs)
        qubit_order: list of integers representing the order of qubits (initially [0, 1, ..., n-1])

        Note that
        - Qubit indexing is always 0-based.
        - Due to the possibility of column swaps, which are used in the algorithm to find a diagonalization of X 
          (algorithm 1), the qubit order stays not always the same.
          However, for the tableau, where only the physical qubits are applied, the qubit order can not change.
          Therefore one has to find a convention, how to label gate operations in the gate_log:
            - An element of a gate log list is always a tuple of the form (gate_name, qubit_indices).
            - The qubit indices in gate_log are the physical qubit indices and not necessarily the numbers of the 
              columns it is applied to in the tableau
            - The qubit indices in the gate_log_column_indexing are the column indices of the tableau

        """
        self.X = X.copy()  # m x n binary matrix
        self.Z = Z.copy()  # m x n binary matrix
        self.s = s.copy()  # m-vector of signs
        self.m, self.n = X.shape

        # gate_log stores the operations applied to the tableau
        self.gate_log = []

        # For difference between gate_log_column_indexing and gate_log see comment above
        self.gate_log_column_indexing = []

        # Maintain the qubit ordering; initially qubit i is in position i
        self.qubit_order = list(range(self.n))
    
    def __str__(self):
        """
        Return a human-readable string representation of the tableau.
        Calling print(tableau_instance), this method is called to display tablau_instance.
        """
        rep = "Tableau:\n"
        rep += "X =\n" + str(self.X) + "\n"
        rep += "Z =\n" + str(self.Z) + "\n"
        rep += "s = " + str(self.s) + "\n"
        rep += "Qubit order: " + str(self.qubit_order) + "\n"
        return rep

    # ----- Virtual operations on the tableau (not physical gates/operations) -----
    def row_swap(self, i, j):
        """
        Swap rows i and j in X, Z, and s.
        """
        if i == j:
            return
        self.X[[i, j], :] = self.X[[j, i], :]
        self.Z[[i, j], :] = self.Z[[j, i], :]
        self.s[[i, j]] = self.s[[j, i]]

        # Log the row-swap in the gate_log (although only virtual operation)
        self.gate_log.append(("row_swap", i, j)) 
        self.gate_log_column_indexing.append(("row_swap", i, j))
    
    def col_swap(self, i, j):
        """
        Swap columns i and j in X and Z and update qubit order.
        """
        if i == j:
            return
        self.X[:, [i, j]] = self.X[:, [j, i]]
        self.Z[:, [i, j]] = self.Z[:, [j, i]]

        # Log the column-swap in the gate_log (although only virtual operation)
        self.gate_log.append(("col_swap", self.qubit_order[i], self.qubit_order[j])) 
        self.gate_log_column_indexing.append(("col_swap", i, j))

        # Update qubit order accordingly
        self.qubit_order[i], self.qubit_order[j] = self.qubit_order[j], self.qubit_order[i]
        
    def row_sweep(self, target, source):
        """
        Sweep (add) row 'source' into row 'target' modulo 2.
        This corresponds to multiplying the Pauli represented by target by the one in source.
        (Sign update is ommited here)
        """
        # Add (mod2) the rows in X and Z:
        self.X[target, :] = (self.X[target, :] + self.X[source, :]) % 2
        self.Z[target, :] = (self.Z[target, :] + self.Z[source, :]) % 2
        
        # We do not implement the sign update here as this will not be used when diagonalising
        # the actual terms of the Hamiltonian
        
        # Log the row_sweep in the gate_log (although only virtual operation)
        self.gate_log.append(("row_sweep", target, source))
        self.gate_log_column_indexing.append(("row_sweep", target, source))
    

    # ----- Physical operations on the tableau that correspond to Clifford gates in the circuit -----
    def apply_H(self, q):
        """
        Apply Hadamard to qubit (column) q.
        It swaps column q in X and Z.
        """    
        self.X[:, q], self.Z[:, q] = self.Z[:, q].copy(), self.X[:, q].copy()
        
        # Sign update
        self.s[:] = (self.s[:] + self.X[:, q] * self.Z[:, q]) % 2

        # Log in gate_log
        self.gate_log.append(("H", self.qubit_order[q]))
        self.gate_log_column_indexing.append(("H", q))
    
    def apply_S(self, q):
        """
        Apply phase gate S to qubit (column) q.
        It adds column q of X to column q of Z (mod 2) and may update signs.
        """        
        self.Z[:, q] = (self.Z[:, q] + self.X[:, q]) % 2

        # Sign update
        self.s[:] = (self.s[:] + self.X[:, q]*self.Z[:, q]) % 2
        
        # Log in gate_log.
        self.gate_log.append(("S", self.qubit_order[q]))
        self.gate_log_column_indexing.append(("S", q))
    
    def apply_CX(self, control, target):
        """
        Apply CNOT gate with control and target qubits.
        It adds column 'control' to column 'target' in X, and column 'target' to column 'control' in Z (both mod 2).
        """
        self.s[:] = (self.s[:] + self.X[:, control] * self.Z[:, target] * 
                     ((1 + self.X[:, target] + self.Z[:, control]) % 2)) % 2
        
        self.X[:, target] = (self.X[:, target] + self.X[:, control]) % 2
        self.Z[:, control] = (self.Z[:, control] + self.Z[:, target]) % 2
        self.gate_log.append(("CX", self.qubit_order[control], self.qubit_order[target]))
        self.gate_log_column_indexing.append(("CX", control, target))
    
    def apply_CZ(self, control, target):
        """
        Apply CZ gate on control and target qubits
        One way to do this is: H(b) CX(a, b) H(b).
        """
        self.apply_H(target)
        self.apply_CX(control, target)
        self.apply_H(target)

        # Delete three last logs in gate_log, as they are generated by applying three gates in a row to compute CZ
        del self.gate_log[-3:]
        # Log the CZ operation in gate_log
        self.gate_log.append(("CZ", self.qubit_order[control], self.qubit_order[target]))
        self.gate_log_column_indexing.append(("CZ", control, target))
    
    # ----- Diagonalization algorithms -----

    def diagonalize_X_block(self):
        """
        Diagonalize the X block of the tableau (Algorithm 1 from [van den Berg and Temme, 2020])
        """

        # Stage 1 (Lines 1-12 in Algo. 1): Diagonalize X block.
        k = 0
        m, n = self.m, self.n
        while True:
            found = False
            # Look for an element 1 in the X block with row i>=k and column j>=k.
            for i in range(k, m):
                for j in range(k, n):
                    if self.X[i, j] == 1:
                        # Bring the 1 to position (k,k) by swapping rows and columns.
                        self.row_swap(i, k)
                        self.col_swap(j, k)
                        # Now eliminate all other 1's in column k.
                        for i2 in range(m):
                            if i2 != k and self.X[i2, k] == 1:
                                self.row_sweep(i2, k)
                        k += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        # kx is stored for Stage 3 of the algorithm
        kx = k
        # Log the completion of Stage 1 (for debugging)
        self.gate_log.append(("Algo. 1, lines 1-12 completed", "k=kx={}".format(kx)))
        self.gate_log_column_indexing.append(("Algo. 1, lines 1-12 completed", "k=kx={}".format(kx)))

        # Stage 2 (Lines 13-22 in Algo. 1): Diagonalize the Z block on the submatrix starting at (k,k).
        while True:
            found = False
            for i in range(k, m):
                for j in range(k, n):
                    if self.Z[i, j] == 1:
                        self.row_swap(i, k)
                        self.col_swap(j, k)
                        # Sweep all other rows that have a 1 in column k of Z.
                        for i2 in range(m):
                            if i2 != k and self.Z[i2, k] == 1:
                                self.row_sweep(i2, k)
                        k += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        # Log the completion of Stage 2 (for debugging)
        self.gate_log.append(("Algo. 1, lines 13-22 completed", "k={}".format(k)))
        self.gate_log_column_indexing.append(("Algo. 1, lines 13-22 completed", "k={}".format(k)))

        
        # Stage 3 (Lines 23-25 in Algo. 1): For each column j in [kx, k), apply Hadamard gate.
        for j in range(kx, k):
            self.apply_H(j)
        
        # Stage 4 (Lines 26-28 in Algo. 1): Clear any remaining 1's in X that lie to the right of the k×k block.
        for i in range(0, k):
            for j in range(k, n):
                if self.X[i, j] == 1:
                    self.apply_CX(i, j)
        # Log the completion of the algorithm (for debugging)
        self.gate_log.append(("Algo. 1 completed"))
        self.gate_log_column_indexing.append(("Algo. 1 completed"))

        # The final k is the rank of the [X,Z] tableau
        return k

    def update_Z_pairwise_clear_X(self, k):
        """
        Update the Z block (and clear X) by pairwise elimination.
        This implements Algorithm 2 from [van den Berg and Temme, 2020]
        k is the rank of the [X,Z] tableau received as output from Algorithm 1.
        """
        # First, for each i from 1 to k-1 (0-indexed), eliminate off-diagonal Z entries (Lines 1-4 in Algo. 2)
        for i in range(1, k):
            for j in range(i):
                if self.Z[i, j] == 1:
                    self.apply_CZ(i, j)

        # Then, for each i in 0 to k-1, adjust the diagonal (Lines 5-9 in Algo. 2)
        for i in range(k):
            if self.Z[i, i] == 1:
                self.apply_S(i)
            self.apply_H(i)

        # Log the completion of Algorithm 2 (for debugging)
        self.gate_log.append(("Algo. 2 completed"))
        self.gate_log_column_indexing.append(("Algo. 2 completed"))
    
    def full_diagonalization(self):
        """
        A full routine to clear the X block of an (allowed) initial tableau and therefore diagonalize the
        corresponding Hamiltonian.
        """
        # Algorithm 1: Diagonalize the X block
        k = self.diagonalize_X_block()
        # Algorithm 2: Clear the X block
        self.update_Z_pairwise_clear_X(k)
        # Return the rank
        return k

    # ----- Extract and apply only physical gates from a given gate_log-----
    def apply_only_clifford_gates(self, input_gate_log):
        """
        Receive a full gate_log and only apply the physical operations ("H", "S", "CX", "CZ") to the tableau.
        It is important, that input_gate_log is a gate log which uses the indexing of the physical qubits,
        i.e. a gate log, which one receives from [tableau object].gate_log and NOT from
        [tableau object].gate_log_column_indexing.
        """
        # Iterate through the gate_log, which is a list of tuples (gate_name, qubit(s) ).
        for op in input_gate_log:
            op_name = op[0]

            # Now process physical (Clifford) gates.
            # Single qubit gates
            if op_name in ("H", "S"):
                # Single-qubit gates.
                q = op[1]
                if op_name == "H":
                    self.apply_H(q)
                else:
                    self.apply_S(q)

            # Two-qubit gates.
            elif op_name in ("CX", "CZ"):
                control, target = op[1], op[2]
                if op_name == "CX":
                    self.apply_CX(control, target)
                else:
                    self.apply_CZ(control, target)
                
            # For any log message or other virtual operations, we skip.
            else:
                continue
    
    # ----- Simplify the Z tableau in a kind of gaussian elimination -----
    def simplify_Z(self):
        """
        Algorithm, which is introduced in the accompanying paper and assumes to be applied to a tableau which was
        already simplified by Algorithms 1 and 2.
        In short, it implements a kind of Gaussian elimination on the Z tableau (the X block is already cleared),
        such that in the end most of the rows are cleared up to a single 1. 
        """
        qubits_unused = list(range(0,self.n))
        gate_log_simplify = []

        # Iterate over each row in the Z matrix
        for row in range(self.m):
            # Iterate over the unused qubits
            for column in qubits_unused:  
            # Identify positions in the Z matrix where a 1 exists to perform elimination
                if self.Z[row, column] == 1:
                    # Remove the current column from the list of unused qubits
                    qubits_unused = [x for x in qubits_unused if x != column]
                    # Iterate over all columns to eliminate other 1's in the same row
                    for column_delete in range(self.n):
                    # Skip the current column and check if there is a 1 in the Z matrix
                        if column_delete != column and self.Z[row, column_delete] == 1:
                            # Apply a CX gate to clear the 1 in the Z matrix
                            self.apply_CX(column_delete, column)
                            gate_log_simplify.append(("CX", column_delete, column))
                    break

        return gate_log_simplify

    # -------- Functions to visualize the tableau/parts of it --------
    def print_tableaus_X_Z(self): 
        """
        Print the X and Z binary matrices and the sign vector alongside each other.
        """
        # Print headers for the matrices
        print("X".ljust(self.n * 2) + "     " + "Z")
        for x_row, z_row, s_val in zip(self.X, self.Z, self.s):
            # Convert rows to string representations with a space between each element
            x_str = " ".join(map(str, x_row))
            z_str = " ".join(map(str, z_row))
            s_str = str(s_val)               # no join needed here
            print(f"{x_str}      {z_str}      {s_str}")

    def print_final_interactions(self):
        """
        Convert a tableau back into a mathematical expression, i.e. a sum of Pauli operators 
        (±X_j, ±Z_j, ±Y_j are the corresponding Pauli matrices acting on qubit j).
        """
        print("\nFinal interactions:")
        # Iterate over the rows of the tableau (one row corresponds to one term in the Hamiltonian)
        for i in range(self.m):
            # Print the sign of the term
            if self.s[i] == 0: print("+", end=" ")
            else: print("-", end=" ")
            # Depending on the entries in this row of the tableau, print the corresponding Pauli operators 
            for j in range(self.n):
                if self.X[i,j]==0 and self.Z[i, j] == 1:
                    print("Z_{}".format(j), end=" ")
                elif self.X[i,j]==1 and self.Z[i, j] == 0:
                    print("X_{}".format(j), end=" ")
                elif self.X[i,j]==1 and self.Z[i, j] == 1:
                    print("Y_{}".format(j), end=" ")
        print("\n")

#-------- Function to check if the processed tableaus are in the form expected ------------                  
    def check_final_Z_tableau(self, model):
        """
        After all the algorithms (Algo. 1, Algo. 2 and the modified gaussian elimination "simplfiy_Z()") have been
        applied to a tableau, which initially corresponded to a Hamiltonian of a given interaction model 
        (e.g. 2D toric code, Haah's code, etc), check if the Z tableau is in the correct final form proposed in the
        accompanying paper for the different models.
        """

        ### Helper function to check sub block ###
        def check_block_identity(cols, start, end, exclude_rows=[]):
            """
            Check if the sub-block of the tableau of all rows in range(start, end+1) (i.e. indluding the row with 
            index end) and columns in cols is an identity matrix but possibly with wrong column order.
            Further it is ensured that all rows which are not in range(start, end+2) have only zeros in the columns 
            in cols (as long as there are no rows which should explicitly not be checked given by the optional
            parameter exclude_rows) and that the rows in range(start, end+1) have only zeros in the columns 
            which are not in cols.

            Note that the row with index end+1 is not checked to be zero for all columns in cols - this row 
            actually holds many 1s for almost all models

            Returns: True if the conditions abover are fulfilled, False otherwise.
            """
            # Check that each column j in cols has exactly one single 1 in the rows in range(start, end+1)
            # (i.e. the rows in {start, start+1, ..., end})
            col_counts = self.Z[start:end+1, cols].sum(axis=0)
            if not np.all(col_counts == 1):
                return False

            # Check that each row i in range(start, end+1) has exactly one 1 across the columns in cols
            row_counts = self.Z[start:end+1, cols].sum(axis=1)
            if not np.all(row_counts == 1):
                return False

            # Ensure zeros in every row before start and after end+1 for columns in cols.            
            if start > 0 or end+1 < self.n-1:
                rows_to_check = np.setdiff1d(np.arange(self.m), np.union1d(np.arange(start, end+2), exclude_rows))
                if np.any(self.Z[np.ix_(rows_to_check, cols)] != 0):
                    return False

            return True

        ##### Check if the X block and the s array is zero #####
        if np.any(self.X != 0):
            raise ValueError("X block is not zero.")
        if np.any(self.s != 0):
            raise ValueError("s array is not zero.")
        
        ##### Find lattice size L from tableau size differently depending on the model #####
        if model == "toric_code":        
            L = round((self.n/2)**0.5)
            if 2*L**2 != self.n or self.n != self.m:
                raise ValueError("Number of qubits or terms doesnt match the 2D toric code arrangement")

        elif model == "haahs_code":
            L = round((self.n/2)**(1/3))
            if 2*L**3 != self.n or self.n != self.m:
                raise ValueError("Number of qubits or terms doesnt match the Haah's code arrangement")

            # In order to get the duality, L has to be odd and not a multiple of 4^p - 1 for some p >= 2.
            if L % 2 == 0:
                raise ValueError("L has to be odd (and not a multiple of 4^p - 1 for some p >= 2).")
            for p in range(2, max(2, math.ceil(math.log2(L-1)/2))):
                if L % (4**p - 1) == 0:
                    raise ValueError("L has to not a multiple of 4^p - 1 for some p >= 2.")
                    
        elif model == "3D_toric":
            L= round((self.n/3)**(1/3))
            L_temp = round((self.m/4)**(1/3))
            if 3*L**3 != self.n or 4*L_temp**3 != self.m:
                raise ValueError("Number of qubits or terms doesnt match the 3D toric code arrangement")

        elif model == "color_honeycomb":
            L = round(((self.n/2)**0.5 - 2)/2)
            if 8*(L+1)**2 != self.n or self.n != self.m:
                raise ValueError("Number of qubits or terms doesnt match the color honeycomb arrangement")
            
        elif model == "stabilizer_subsystem_toric":
            L= round((self.n/3)**(1/3))
            L_temp = round((self.m)**(1/3))

            # L has to be even
            if L % 2 == 1 or 3*L**3 != self.n or L_temp**3 != self.m:
                raise ValueError("Number of qubits or terms doesnt match the toric cube subsystem code arrangement")
        
        elif model == "checks_subsystem_toric":
            L= round((self.n/3)**(1/3))
            L_temp = round((self.m/4)**(1/3))

            # L has to be even
            if L % 2 == 1 or 3*L**3 != self.n or 4*L_temp**3 != self.m:
                raise ValueError("Number of qubits or terms doesnt match the toric triangle subsystem code arrangement")

        elif model == "X_cube":
            L = round(((2/17)*(self.m + 1.5 * self.n))**(1/3))
            if 4*L**3 - 3*L**2 != self.m or 3*L**3 + 2*L**2 != self.n:
                raise ValueError("Number of qubits or terms doesnt match the X cube arrangement")

        elif model == "rotated_surface": 
            L = round((self.n)**0.5-1)
            if (L+1)**2 != self.n or L**2+2*L != self.m:
                raise ValueError("Number of qubits or terms doesnt match the rotated surface code arrangement")
            
        else:
            raise ValueError("Unknown model: {}".format(model))
        

        ##### Check how many columns in the Z block are completely zero #####
        zero_cols = list(np.where(self.Z.sum(axis=0) == 0)[0])
        nr_zero_cols = len(zero_cols)
        log_message("Number of zero columns: " + str(nr_zero_cols))

	
        ##### Now with the preliminary work from above, check the tableau differently depending on the model #####
        # (for the specified form of the tableau depending on the model for which is checked here,
        # see also the appendix of the accompanying paper)
        if model == "toric_code":

            # Toric code in 2D should have 2 zero columns
            if nr_zero_cols != 2:
                return False

            # Rows L**2-1 and 2L**2-1 (remember indexing starts from 0) should have L**2-1 many 1's each with no overlap
            full_row_1 = list(np.where(self.Z[L**2-1, :] == 1)[0])
            full_row_2 = list(np.where(self.Z[2*L**2-1, :] == 1)[0])
            
            # Check that there is no overlap
            if len(set(full_row_1).intersection(full_row_2)) != 0 or len(full_row_1) != L**2-1 or len(full_row_2) != L**2-1:
                return False

            # Check that the two subsets of columns are identity matrices (possibly with wrong column order)
            if not check_block_identity(full_row_1, 0, L**2-2):
                return False
            if not check_block_identity(full_row_2, L**2, 2*L**2-2):
                return False

            # If all these checks pass, the tableau is in the correct final form
            return True
        

        elif model == "haahs_code":

            # Haah's code (in 3D) should have 2 zero columns
            if nr_zero_cols != 2:
                return False

            # Rows L**3-1 and 2L**3-1 (remember indexing starts from 0) should have L**3-1 many 1's each with no overlap
            full_row_1 = list(np.where(self.Z[L**3-1, :] == 1)[0])
            full_row_2 = list(np.where(self.Z[2*L**3-1, :] == 1)[0])
            
            # Check that there is no overlap
            if (len(set(full_row_1).intersection(full_row_2)) != 0 or len(full_row_1) != L**3-1 
                or len(full_row_2) != L**3-1):
                return False

            # Check that the two subsets of columns are identity matrices (possibly with wrong column order)
            if not check_block_identity(full_row_1, 0, L**3-2):
                return False
            if not check_block_identity(full_row_2, L**3, 2*L**3-2):
                return False

            # If all these checks pass, the tableau is in the correct final form
            return True
        
        
        elif model == "3D_toric":

            # Toric code (in 3D) should have 3 zero columns
            if nr_zero_cols != 3:
                return False

            # Row L**3-1 (remember indexing starts from 0) should have L**3-1 many 1's
            full_row_1 = list(np.where(self.Z[L**3-1, :] == 1)[0])
            
            # Check that there are exactly L**3-1 many 1's
            if len(full_row_1) != L**3-1:
                return False

            # Check that the subset of columns full_row_1 includes an identity matrix (possibly with wrong column order)
            if not check_block_identity(full_row_1, 0, L**3-2):
                return False
            
            # Ensure that all columns not included in full_row_1 (identity block) or zero_cols (completely zero columns)
            # have non-zero entries only in rows corresponding to L**3 and higher
            for i in range(self.n):
                if i not in full_row_1 and i not in zero_cols:
                    if np.any(self.Z[:L**3, i] != 0):
                        return False

            # If all these checks pass, the tableau is in the correct final form
            return True
        
        
        elif model == "color_honeycomb":
            
            # In case L mod 3 = 2 the tableau should have 4 zero columns and be made up out of two subblock identities
            # with each having two lines of 1s and 0s below
            if L % 3 == 2:
                if nr_zero_cols != 4:
                    return False
                
                # Rows L**3-1 and 2L**3-1 (remember indexing starts from 0) should have L**3-1 many 1's each 
                # with no overlap
                full_row_1 = list(np.where(np.logical_or(self.Z[4*(L+1)**2-1, :] == 1, self.Z[4*(L+1)**2-2, :] == 1))[0])
                full_row_2 = list(np.where(np.logical_or(self.Z[8*(L+1)**2-1, :] == 1, self.Z[8*(L+1)**2-2, :] == 1))[0])
                
                # Check that there is no overlap
                if (len(set(full_row_1).intersection(full_row_2)) != 0 or len(full_row_1) != 4*(L+1)**2-2 
                    or len(full_row_2) !=  4*(L+1)**2-2):
                    return False

                # Check that the two subsets of columns are identity matrices (possibly with wrong column order)
                # Note that for the color honeycomb model, there are two possible non-zero lines after the 
                # identity block, therefore add the second row to exclude_rows.
                if not check_block_identity(full_row_1, 0, 4*(L+1)**2-3, exclude_rows=[4*(L+1)**2-1]):
                    return False
                if not check_block_identity(full_row_2, 4*(L+1)**2, 8*(L+1)**2-3, exclude_rows=[8*(L+1)**2-1]):
                    return False

                # If all these checks pass, the tableau is in the correct final form
                return True
            
            # In case L mod 3 = 0,1 the tableau should be exactly the identity matrix (i.e. non-interacting matrix)
            else:
                if nr_zero_cols != 0:
                    return False
                
                # Check that it's the identity matrix
                return check_block_identity(list(range(0,8*(L+1)**2)), 0, 8*(L+1)**2 -1)
            
        elif model == "stabilizer_subsystem_toric":

            # the model should have 2L**3+2 zero columns
            if nr_zero_cols != 2*L**3+2:
                return False

            # Rows (L**3)/2-1 and L**3-1 (remember indexing starts from 0) should have (L**3)/2-1 many 1's each 
            # with no overlap
            full_row_1 = list(np.where(self.Z[int((L**3)/2-1), :] == 1)[0])
            full_row_2 = list(np.where(self.Z[int(L**3-1), :] == 1)[0])
            
            # Check that there is no overlap
            if (len(set(full_row_1).intersection(full_row_2)) != 0 or len(full_row_1) != (L**3)/2-1 
                or len(full_row_2) != (L**3)/2-1):
                return False

            # Check that the two subsets of columns are identity matrices (possibly with wrong column order)
            if not check_block_identity(full_row_1, 0, int((L**3)/2-2)):
                return False
            if not check_block_identity(full_row_2, int((L**3)/2), int(L**3-2)):
                return False

            # If all these checks pass, the tableau is in the correct final form
            return True
        
        
        elif model == "checks_subsystem_toric":

            # the model should have no zero columns
            if nr_zero_cols != 0:
                return False
            
            # There should be exactly L**3 rows with three 1s and 3*L**3 rows with one 1
            row_sums = self.Z.sum(axis=1)
            full_rows_3 = list(np.where(row_sums == 3)[0])
            full_rows_1 = list(np.where(row_sums == 1)[0])
            if len(full_rows_3) != L**3 or len(full_rows_1) != 3*L**3:
                return False
            
            # Check that there is no overlap between the different rows with 3 entries
            col_counts_3 = self.Z[full_rows_3,:].sum(axis=0)
            if not np.all(col_counts_3 == 1):
                return False
            
            # Check that there is no overlap between the different rows with 1 entry
            col_counts_1 = self.Z[full_rows_1,:].sum(axis=0)
            if not np.all(col_counts_1 == 1):
                return False
            
            # If all these checks pass, the tableau is in the correct final form
            return True
        

        elif model == "X_cube":

            ### The model should have 4*L**2+2*L-1 zero columns
            if nr_zero_cols != 4*L**2+2*L-1:
                return False

            ### Part 1
            # From the top of the tableau, check L identity blocks with each having L**2 rows and L**2-1 columns 
            # (including the row of 1s on the bottom)
            # Build the list of row-indices
            rows1 = [(i+1)*L**2 - 1 for i in range(L)]
            # For each such row, grab all the column-indices where Z[row, col] == 1
            full_rows_part1 = [list(np.where(self.Z[row, :] == 1)[0]) for row in rows1]

            # Check that each of the rows has L**2-1 many 1's
            for i in range(L):
                if len(full_rows_part1[i]) != L**2-1:
                    return False
            
            # Check that the corresponding subsets of columns are identity matrices (possibly with wrong column order)
            for i in range(L):
                if not check_block_identity(full_rows_part1[i], i*L**2, (i+1)*L**2-2):
                    return False

            ### Part 2
            # Check L-1 blocks, each made up out of L**2-1 identity matrices of size 2x2 with each having one row of 
            # two 1s directly below as well as special three rows on the bottom of the block
            for j in range(L-1):
                row_anker = L**3 + j*3*L**2 -1

                # Inside each block, check that there are L**2-1 rows with two 1's each and that the two lines above 
                # are an identity (possibly with wrong column order)
                for row_add in range(3, 3*L**2, 3):
                    row_nr = row_anker + row_add
                    full_row = list(np.where(self.Z[row_nr, :] == 1)[0])

                    if len(full_row) != 2:
                        return False
                    
                    if not check_block_identity(full_row, row_nr-2, row_nr-1,
                                                exclude_rows=[row_anker+3*L**2-1, row_anker+3*L**2]):
                        return False
                    # To check the correct structure of the last two rows of the block use the following conditions:
                    if not ((self.Z[row_nr-1, full_row[0]] == 1 and 
                            self.Z[row_anker+3*L**2-1, full_row[0]] == 1 and
                            self.Z[row_anker+3*L**2, full_row[0]] == 1 and
                            self.Z[row_anker+3*L**2-1, full_row[1]] == 0 and
                            self.Z[row_anker+3*L**2, full_row[1]] == 0) ^
                            (self.Z[row_nr-1, full_row[1]] == 1 and
                            self.Z[row_anker+3*L**2-1, full_row[1]] == 1 and
                            self.Z[row_anker+3*L**2, full_row[1]] == 1 and
                            self.Z[row_anker+3*L**2-1, full_row[0]] == 0 and
                            self.Z[row_anker+3*L**2, full_row[0]] == 0)):
                        return False
                    
                # Check that the third last row of the block has only a single 1 and that the two entries
                # below this entry are 0 and 1 
                single_column = list(np.where(self.Z[row_anker+3*L**2-2, :] == 1)[0])
                if len(single_column) != 1:
                    return False
                if not (self.Z[row_anker+3*L**2-1, single_column[0]] == 0 and
                        self.Z[row_anker+3*L**2, single_column[0]] == 1):
                    return False

            ### If all these checks pass, the tableau is in the correct final form
            return True
        
        elif model == "rotated_surface":

            # The model should have one zero column
            if nr_zero_cols != 1:
                return False
            # The remaining matrix (without the zero column) should be an identity matrix of size (L+1)**2-1 x (L+1)**2-1
            if not check_block_identity(list(range(0, self.n-1)), 0, (L+1)**2-2):
                return False
                
            # If these checks pass, the tableau is in the correct final form
            return True
        
        else:
            raise ValueError("Unknown model: {}".format(model))


def check_matrix_sums(matrix):
    """
    Given a matrix, adds all its values in a column and row-wise fashion, and returns the maximum row sum and the maximum column sum

    Input:
        matrix (np.ndarray): The input NumPy matrix with 0s and 1s.

    Returns:
        int, int: max row sum, max col sum
    """
    # Calculate the sum of each row
    row_sums = np.sum(matrix, axis=1)
    
    # Calculate the sum of each column
    col_sums = np.sum(matrix, axis=0)
    
        
    return np.max(row_sums), np.max(col_sums)
        
# ------------------------- End of Tableau class definition-------------------------------





#------------------------
# Helper Functions (for documenting checks and constructing tableaus)
#-------------------------

def log_message(message, file_path="output.txt"):
    """
    Helper function to document the systematic checking of diagonalized tableaus.
    Prints the message to console, then appends it to file_path.
    Flushes & fsyncs so the data is guaranteed to be on disk after each call.
    """
    # Print to console
    print(message)

    # Append to file, flush & fsync
    with open(file_path, "a") as f:
        f.write(message + "\n")
        f.flush()              # flush Python’s internal buffer
        os.fsync(f.fileno())   # tell OS to flush its buffers to disk

# --- Helper functions to construct tableaus for different models ---
def onedposhaah(L, c, r, l): 
    """
    Function that associates a non-negative integer to each coordinate in 3D used in the Haah lattice, 
    assuming periodic boundary conditions. Note that in the Haah model, there are two spins per vertex in the lattice.

    Input: Integers c, r, l representing the column, row and layer of a spin, respectively. 
           Integer L representing the number of columns and rows of the lattice 
    
    Returns: Integer that identifies the spin in such position. 
    """
    ret = c + 2 * L * r + 2 * (L**2) * l
    
    return ret

def onedpostoric(L, c, r, l): 
    """
    Function that associates a non-negative integer to each coordinate in 3D used in the 3D toric code. 
    Note that in the toric code, the spins are placed in the edges of the lattice.

    Input: Integers c, r, l representing the column, row and layer of a spin, respectively. 
           Integer L representing the number of cubes in an edge of the lattice 
           (e.g. if L = 2 then there are 2^3 cubes in total)

    Returns: Integer that identifies the spin in such position
    """
    spins_f = (2 * L**2) #Number of spins in each layer containing the horizontal faces of the cubes
    spins_e = L**2 #Number of spins in each layer containing the vertical edges of the cubes
    layers_f = (l // 2 + l % 2) #Number of layers containing the horizontal faces of the cubes under the current layer
    layers_e = (l // 2) #Number of layers containing the vertical edges of the cubes under the current layer
    
    ret = c + L*r + spins_f*layers_f + spins_e*layers_e
    
    return ret

def onedposXcube(L, c, r, l):
    """
    Function that associates a non-negative integer to each coordinate in 3D used in the X-cube model, 
    assuming cylindrical boundary conditions. 
    Note that in the X cube model, the spins are placed in the edges of the lattice

    Input: Integers c, r, l representing the column, row and layer of a spin, respectively. 
           Integer L representing the number of cubes in an edge of the lattice (e.g. if L = 2 then there are 2^3 cubes in total)

    Returns: Integer that identifies the spin in such position
    """
    spins_f = (L + 1) * L + L * L #Number of spins in each layer containing the horizontal faces of the cubes
    spins_e = (L + 1) * L #Number of spins in each layer containing the vertical edges of the cubes
    layers_f = (l // 2 + l % 2)  #Number of layers containing the horizontal faces of the cubes under the current layer
    layers_e = (l // 2) #Number of layers containing the vertical edges of the cubes under the current layer

    if l % 2 == 0: 
        ret = c + (r // 2 + r % 2) * L + (r // 2) * (L + 1) + spins_f*layers_f + spins_e*layers_e
    else: 
        ret = c + r * (L + 1) + spins_f*layers_f + spins_e*layers_e
    
    return ret

def onedpossurface(L, x, y): 
    """
    Function that associates a non-negative integer to each coordinate in the 2D rotated surface code,
    assuming open boundary conditions.
    """
    return x + y*(L+1)

# -------------------------------
# Model-Specific Tableau Builders
# -------------------------------

def toric_code_matrices(L):
    """
    Constructs binary matrices representing the toric code stabilizers on an L×L torus.
    The toric code is defined on a square lattice with periodic boundary conditions.

    Input: Lattice size parameter L (positive integer)

    Returns:
    X, Z: two numpy arrays of shape (2*L^2, 2*L^2).
        X has the X–type (star) operators in its first L^2 rows (the rest are zeros),
        and Z has the Z–type (plaquette) operators in its last L^2 rows (the first L^2 are zeros).
    s: a numpy array with zeros of shape (2*L^2,) representing the sign vector with all signs positive.

    The qubits are labeled from 0 to 2*L^2-1 in the following way:
    --0----1----2--...--L-1--
    |    |    |    ...       |
    L   L+1  L+2   ...     2L-1
    |    |    |    ...       |
    --2L--2L+1--2L+2--...--3L-1
    :
    :
    (The lines represent the plaquette terms, the star terms are arranged accordingly)
    
    ##### Example: Toric code on 8 qubits (i.e. L=2):
      X = np.array([[1, 1, 1, 0, 0, 0, 1, 0],
                    [1, 1, 0, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
    ])
      Z = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 0, 0],
                    [1, 0, 1, 1, 1, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 1, 1],
                    [0, 1, 0, 0, 0, 1, 1, 1],
    ])
        # Sign vector s (0 means positive, 1 negative)
        s = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    """
    # Total number of qubits
    N = 2 * L * L  

    # Initialize the X and Z matrices and the sign vector
    X = np.zeros((N, N), dtype=int)
    Z = np.zeros((N, N), dtype=int)
    s = np.zeros(N, dtype=int)  # sign vector (0 for +)

    # Counter to keep track of the number of operators added
    counter = 0

    # Build star (vertex) operators (X-type)
    for i in range(L):
        for j in range(L):

            # Each star operator involves 4 qubits (indices):
            star_left = 2*i*L + j
            star_right = 2*i*L + ((j + 1) % L) 
            star_down = (2*i+1)*L + ((j + 1) % L)
            if i==0:
                star_up = (2*L-1) * L + ((j + 1) % L)
            else:
                star_up = (2*i - 1) * L + ((j + 1) % L)

            # Represent the operator in one row of the tableau
            X[counter, star_left] = 1
            X[counter, star_right] = 1
            X[counter, star_down] = 1
            X[counter, star_up] = 1
            counter += 1

    # Build plaquette (face) operators (Z-type)
    for i in range(L):
       for j in range(L):
                
                # Each star operator involves 4 qubits (indices):
                plaquette_up = 2*i*L + j
                plaquette_left = (2*i+1)*L + j
                plaquette_right = (2*i+1)*L + ((j + 1) % L)
                if i == L-1:
                    plaquette_down = j
                else:
                    plaquette_down = (2*i+2)*L + j

                # Represent the operator in one row of the tableau
                Z[counter, plaquette_up] = 1
                Z[counter, plaquette_left] = 1
                Z[counter, plaquette_right] = 1
                Z[counter, plaquette_down] = 1
                counter += 1
    
    return X, Z, s


def color_honeycomb_matrices(L):
    """
    Constructs binary matrices representing the color code on a honeycomb lattice with periodic boundary conditions.
    
    Input: Integer L>= 0, which specifies the dimensions of the lattice
    (regarding dimensions of lattice depending on L, see also comments below)

    Returns: X,Z,s

    The qubits are labeled from 0 to N-1 in the following way:
    0           2           4                ...
          1          3             5         ...

        L_1+1      L_1+3         L_1+5       ...
    L_1       L_1+2       L_1+4              ...

    2*L_1    2*L_1+2      2*L_1+4            ...
        2L_1+1     2L_1+3       2L_1+5        ...
    .
    .
    .
    
    """
    
    L_1 = 4*L+4 # horizontal dimension of lattice
    L_2 = 2*L+1 # vertical dimmension of lattice
    # (number of horizontal honeycombs: L_1/2 = 2*L+2)
    # (number of vertical honeycombs: L_2+1 = 2*L+2)
    # The lattice is thefore always a square lattice

    # Total number of qubits
    N = (L_2 + 1) * L_1 
    
    # Initialize the X and Z matrices and the sign vector
    # (for this model number qubits n = number of interaction terms m)
    X = np.zeros((N, N), dtype=int)
    Z = np.zeros((N, N), dtype=int)
    s = np.zeros(N, dtype=int)  # sign vector (0 for +, 1 for -)
    
    # Counter to keep track of the number of operators added
    counter = 0

    # In the following the X and Z plaquette operators are each built (i.e. added as a row in the tableau) first for 
    # the even row indices and then for the odd row indices.
    # (This is done so the final simplified tabeleau has a nicer structure, the ordering of the rows in the tableau
    # has no physical meaning)

    # Build X-plaquette operators for even rows of the honeycomb lattice (i.e. row is an even multiple of L_1)
    for row in range(0, L_2*L_1, 2*L_1): # row: 0,2*L_1, 4*L_1, ..., (L_2 - 1)*2L_1
        for column in range(0, L_1, 2):
            
            # Each X-plaquette operator involves 6 qubits (indices):
            hc_up_up = row + column
            hc_up_right = row + column + 1
            if column == 0:
                hc_up_left = row + L_1 - 1
            else:
                hc_up_left = row + column - 1
            
            hc_down_down = row + column + L_1
            hc_down_right = row + column + L_1 + 1
            if column == 0:
                hc_down_left = row + L_1 - 1 + L_1
            else:
                hc_down_left = row + column - 1 + L_1
            
            # Represent the operator in one row of the tableau
            X[counter, hc_down_left] = 1
            X[counter, hc_up_left] = 1
            X[counter, hc_up_up] = 1
            X[counter, hc_up_right]= 1
            X[counter, hc_down_right] = 1
            X[counter, hc_down_down] = 1
            counter += 1
    
    # Build X-plaquette operators for odd rows of the honeycomb lattice (i.e. row is an odd multiple of L_1)
    for row in range(L_1, L_2*L_1+2*L_1, 2*L_1): # row: L_1,3*L_1, 5*L_1, ..., L_2*L_1
        for column in range(1, L_1, 2):
            
            # Each X-plaquette operator involves 6 qubits (indices):
            hc_up_up = row + column
            hc_up_left = row + column - 1
            if column == L_1 - 1:
                hc_up_right= row
            else:
                hc_up_right = row + column + 1
            
            if row == L_2*L_1:
                hc_down_down = column
            else:
                hc_down_down = row + column + L_1

            hc_down_left = hc_down_down - 1

            if column == L_1 - 1:
                if row == L_2*L_1:
                    hc_down_right = 0
                else:
                    hc_down_right = row + L_1
            else:
                if row == L_2*L_1:
                    hc_down_right = column + 1
                else:
                    hc_down_right = row + column + L_1 + 1
            
            # Represent the operator in one row of the tableau
            X[counter, hc_down_left] = 1
            X[counter, hc_up_left] = 1
            X[counter, hc_up_up] = 1
            X[counter, hc_up_right]= 1
            X[counter, hc_down_right] = 1
            X[counter, hc_down_down] = 1
            counter += 1
    
    # Build Z-plaquette operators for even rows of the honeycomb lattice (i.e. row is an even multiple of L_1)
    for row in range(0, L_2*L_1, 2*L_1): # row: 0,2*L_1, 4*L_1, ..., (L_2 - 1)*2L_1
        for column in range(0, L_1, 2):
            
            # Each Z-plaquette operator involves 6 qubits (indices):
            hc_up_up = row + column
            hc_up_right = row + column + 1
            if column == 0:
                hc_up_left = row + L_1 - 1
            else:
                hc_up_left = row + column - 1
            
            hc_down_down = row + column + L_1
            hc_down_right = row + column + L_1 + 1
            if column == 0:
                hc_down_left = row + L_1 - 1 + L_1
            else:
                hc_down_left = row + column - 1 + L_1

            # Represent the operator in one row of the tableau
            Z[counter, hc_down_left] = 1
            Z[counter, hc_up_left] = 1
            Z[counter, hc_up_up] = 1
            Z[counter, hc_up_right] = 1
            Z[counter, hc_down_right] = 1
            Z[counter, hc_down_down] = 1
            counter += 1
    
    # Build Z-plaquette operators for odd rows of the honeycomb lattice (i.e. row is an odd multiple of L_1)
    for row in range(L_1, L_2*L_1+2*L_1, 2*L_1): # row: L_1,3*L_1, 5*L_1, ..., L_2*L_1
        for column in range(1, L_1, 2):
            
            # Each Z-plaquette operator involves 6 qubits (indices):
            hc_up_up = row + column
            hc_up_left = row + column - 1
            if column == L_1 - 1:
                hc_up_right= row
            else:
                hc_up_right = row + column + 1
            
            if row == L_2*L_1:
                hc_down_down = column
            else:
                hc_down_down = row + column + L_1

            hc_down_left = hc_down_down - 1

            if column == L_1 - 1:
                if row == L_2*L_1:
                    hc_down_right = 0
                else:
                    hc_down_right = row + L_1
            else:
                if row == L_2*L_1:
                    hc_down_right = column + 1
                else:
                    hc_down_right = row + column + L_1 + 1

            # Represent the operator in one row of the tableau
            Z[counter, hc_down_left] = 1
            Z[counter, hc_up_left] = 1
            Z[counter, hc_up_up] = 1
            Z[counter, hc_up_right] = 1
            Z[counter, hc_down_right] = 1
            Z[counter, hc_down_down] = 1
            counter += 1
            
    return X, Z, s


def haah_matrices(L):
    """
    Constructs binary matrices representing Haah's code Hamiltonian on a 3D square lattice with periodic 
    boundary conditions. There are two spins per vertex of the lattice.

    Input: Integers cols, rows, layers, which specify the dimensions of the lattice

    Returns: X,Z,s

    The qubits are labeled in a bottom-up fashion in the following way:

    .
    .
    .
    

    2*rows,2*rows+1    2*rows+2,2*rows+3     ...

 
    0,1                    2,3               ...
    
    
    """
    # Total number of qubits (and interactions)
    N = 2 * L**3
    
    # Initialize the X and Z matrices and the sign vector
    X = np.zeros((N, N), dtype=int)
    Z = np.zeros((N, N), dtype=int)
    s = np.zeros(N, dtype=int)  # Sign vector (0 for +, 1 for -)
    
    # Counter to keep track of the number of operators added
    counter = 0

    # Start by building the B_v operators which consist of Pauli X and identity operators at the corners of one cube
    # (see also corresponding figure in the accompanying paper)
    for l in range(L): 
        for r in range(L):
            for c in range(0, 2*L, 2): # Iterate only over the first spin in each vertex
                
                # Represent the operator in one row of the tableau
                X[counter, onedposhaah(L, c + 1, r, l)] = 1 #Second qubit down bottom left corner
                X[counter, onedposhaah(L, (c + 2) % (2 * L), r, l)] = 1 #First qubit down bottom right corner
                
                X[counter, onedposhaah(L, c, (r + 1) % L, l)] = 1 #First qubit down top left corner
                X[counter, onedposhaah(L, c + 1, (r + 1) % L, l)] = 1 #Second qubit down top left corner
                X[counter, onedposhaah(L, (c + 3) % (2 * L), (r + 1) % L, l)] = 1 #Second qubit down top right corner

                X[counter, onedposhaah(L, c, r, (l + 1) % L)] = 1 #First qubit up bottom left corner
                X[counter, onedposhaah(L, c + 1, (r + 1) % L, (l + 1) % L)] = 1 #Second qubit up top left corner
                X[counter, onedposhaah(L, (c + 2) % (2 * L), (r + 1) % L, (l + 1) % L)] = 1 #First qubit up top right corner
                counter += 1

    # Now build the A_v operators which consist of Pauli Z and identity operators at the corners of one cube
    for l in range(L): 
        for r in range(L):
            for c in range(0, 2*L, 2): #Iterate only over the first spin in each vertex
    
                # Represent the operator in one row of the tableau
                Z[counter, onedposhaah(L, c + 1, r, l)] = 1 #Second qubit down bottom left corner
                Z[counter, onedposhaah(L, (c + 2) % (2 * L), r, l)] = 1 #First qubit down bottom right corner
                
                Z[counter, onedposhaah(L, (c + 3) % (2 * L), (r + 1) % L, l)] = 1 #Second qubit down top right corner

                Z[counter, onedposhaah(L, c, r, (l + 1) % L)] = 1 #First qubit up bottom left corner
                Z[counter, onedposhaah(L, (c + 2) % (2 * L), r, (l + 1) % L)] = 1 #First qubit up bottom right corner
                Z[counter, onedposhaah(L, (c + 3) % (2 * L), r, (l + 1) % L)] = 1 #Second qubit up bottom right corner

                Z[counter, onedposhaah(L, c + 1, (r + 1) % L, (l + 1) % L)] = 1 #Second qubit up top left corner
                Z[counter, onedposhaah(L, (c + 2) % (2 * L), (r + 1) % L, (l + 1) % L)] = 1 #First qubit up top right corner
                counter += 1
                
    return X, Z, s


def toric_3D_matrices(L):
    """
    Constructs binary matrices representing the 3D toric code Hamiltonian on a 3D cubic lattice 
    (torus of L x L x L blocks) with periodic boundary conditions.
    There is one spin per edge of the lattice.

    Input:
    Integer L which specifies the dimensions of the lattice (in number of cubes)
     
    Returns: X,Z,s

    The qubits are labeled in a bottom-up fashion. For each layer, the spins are numbered also in
    a bottom-up fashion 


    .
    .
    .


          2L           2L+1   ...

    L           L+1           ...

          0           1       ...
    
    """

    # Total number of spins
    N = 3*L**3
    # Total number of interactions
    M = 4*L**3 

    # Initialize the X and Z matrices and the sign vector
    X = np.zeros((M, N), dtype=int)
    Z = np.zeros((M, N), dtype=int)
    s = np.zeros(M, dtype=int)  # Sign vector (0 for +, 1 for -)
    
    # Counter to keep track of the number of operators added
    counter = 0

    # Build the A_v operators (star operators) 
    for l in range(0,2*L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
        for r in range(0, 2*L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
            for c in range(L): 
                
                # Represent the operator in one row of the tableau
                X[counter, onedpostoric(L, c, r, l)] = 1 #site
                X[counter, onedpostoric(L, (c - 1) % L, r, l)] = 1 #west
                X[counter, onedpostoric(L, c, r + 1, l)] = 1 #north
                X[counter, onedpostoric(L, c, (r - 1) % (2 * L), l)] = 1 #south
                X[counter, onedpostoric(L, c, r // 2, l + 1)] = 1 #up
                X[counter, onedpostoric(L, c, r // 2, (l - 1) % (2 * L))] = 1 #down
                
                counter += 1

    # Build the three different B_p operators (plaquette operators)
    for l in range(0,2*L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
        for r in range(0, 2*L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
            for c in range(L): 
                
                # Represent each operator in one row of the tableau:

                # Horizontal plaquette
                Z[counter, onedpostoric(L, c, r, l)] = 1 #site
                Z[counter, onedpostoric(L, c, r + 1, l)] = 1 #up left
                Z[counter, onedpostoric(L, (c + 1) % L, r + 1, l)] = 1 #up right
                Z[counter, onedpostoric(L, c, (r + 2) % (2 * L), l)] = 1 #up up

                counter += 1

                # Vertical plaquette (front-facing)
                Z[counter, onedpostoric(L, c, r, l)] = 1 #site
                Z[counter, onedpostoric(L, c, r // 2, l + 1)] = 1 #up left
                Z[counter, onedpostoric(L, (c + 1) % L, r // 2, l + 1)] = 1 #up right
                Z[counter, onedpostoric(L, c, r, (l + 2) % (2 * L))] = 1 #up up

                counter += 1

                # Vertical plaquette (side-facing)
                Z[counter, onedpostoric(L, c, r + 1, l)] = 1 # bottom 
                Z[counter, onedpostoric(L, c, r + 1, (l + 2) % (2 * L))] = 1 # up up 
                Z[counter, onedpostoric(L, c, (r // 2 + 1) % L, l + 1)] = 1 # up right
                Z[counter, onedpostoric(L, c, r // 2, l + 1)] = 1 # up left

                counter += 1
                
    return X, Z, s


def X_cube_matrices(L):
    """
    Constructs binary matrices representing the X-cube Hamiltonian on a 3D cubic lattice with cylindrical boundary
    conditions. There is one spin per edge of the lattice.

    Input: Integer L which specifies the dimensions of the lattice (in number of cubes)
 
    Returns: X,Z,s 

    The qubits are labeled in a bottom-up fashion. For each layer, the spins are numbered also in
    a bottom-up fashion (see toric_3D_matrices)
    """
    # Total number of spins (cylindrical boundary conditions)
    N = 3 * L**3 + 2 * L**2
    # Total number of interactions
    M = L**3 + 3 * (L-1) * L**2

    # Initialize the X and Z matrices and the sign vector
    X = np.zeros((M, N), dtype=int)
    Z = np.zeros((M, N), dtype=int)
    s = np.zeros(M, dtype=int)  # Sign vector (0 for +, 1 for -)
    
    # Counter to keep track of the number of operators added
    counter = 0

    # Build the cube operators A_b of the X-cube model (see also corresponding figure in the accompanying paper) 
    for i in range(L): 
        for l in range(0,2*L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
            for r in range(0, 2*L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
                for c in range(L):
                    
                    if c % L == i:

                        # Represent the operator in one row of the tableau:
                        # Horizontal bottom plaquette
                        X[counter, onedposXcube(L, c, r, l)] = 1 #site
                        X[counter, onedposXcube(L, c, r + 1, l)] = 1 #up left
                        X[counter, onedposXcube(L, c + 1, r + 1, l)] = 1 #up right
                        X[counter, onedposXcube(L, c, (r + 2) % (2 * L), l)] = 1 #up up
        
                        # Horizontal top plaquette
                        X[counter, onedposXcube(L, c, r, (l + 2) % (2 * L))] = 1 #site
                        X[counter, onedposXcube(L, c, r + 1, (l + 2) % (2 * L))] = 1 #up left
                        X[counter, onedposXcube(L, c + 1, r + 1, (l + 2) % (2 * L))] = 1 #up right
                        X[counter, onedposXcube(L, c, (r + 2) % (2 * L), (l + 2) % (2 * L))] = 1 #up up
                        
                        # Spins in the edges
                        X[counter, onedposXcube(L, c, r // 2, l + 1)] = 1 #front left
                        X[counter, onedposXcube(L, c + 1, r // 2, l + 1)] = 1 #front right
                        X[counter, onedposXcube(L, c, (r // 2 + 1) % L, l + 1)] = 1 # back left
                        X[counter, onedposXcube(L, c + 1, (r // 2 + 1) % L, l + 1)] = 1 # back right
        
                        counter += 1

    # Build the cross operators B_v, C_v and D_v of the X-cube model                    
    for i in range(1, L):
        for l in range(0,2*L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
            for r in range(0, 2*L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
                for c in range(1, L): 
    
                    if c % L == i: 

                        # Represent each of the three cross operators of this cube in one row of the tableau:
                        # Horizontal star operator
                        Z[counter, onedposXcube(L, c, r, l)] = 1 #site
                        Z[counter, onedposXcube(L, c - 1, r, l)] = 1 #west
                        Z[counter, onedposXcube(L, c, (r + 1) % (2 * L), l)] = 1 #north
                        Z[counter, onedposXcube(L, c, (r - 1) % (2 * L), l)] = 1 #south
                        counter += 1
        
                        #Vertical sideways star operator
                        Z[counter, onedposXcube(L, c, (r + 1) % (2 * L), l)] = 1 #north
                        Z[counter, onedposXcube(L, c, (r - 1) % (2 * L), l)] = 1 #south
                        Z[counter, onedposXcube(L, c, r // 2, (l + 1) % (2 * L))] = 1 #up
                        Z[counter, onedposXcube(L, c, r // 2, (l - 1) % (2 * L))] = 1 #down
                        counter += 1
        
                        # Vertical frontal star operator
                        Z[counter, onedposXcube(L, c, r, l)] = 1 #site
                        Z[counter, onedposXcube(L, c - 1, r, l)] = 1 #west
                        Z[counter, onedposXcube(L, c, r // 2, (l + 1) % (2 * L))] = 1 #up
                        Z[counter, onedposXcube(L, c, r // 2, (l - 1) % (2 * L))] = 1 #down
                        counter += 1

                
    return X, Z, s
    

def checks_subsystem_toric_matrices(L):
    """
    Constructs binary matrices representing a simplified version of the subsystem toric code Hamiltonian; consisting
    only of commuting triangle interactions. The model is defined on a 3D square lattice with periodic boundary 
    conditions. There is one spin per edge of the lattice

    Input: Integer L which specifies the dimensions of the lattice (in number of cubes)
 
    Returns: X,Z,s

    The qubits are labeled in a bottom-up fashion. For each layer, the spins are numbered also in
    a bottom-up fashion (see toric_3D_matrices)
    """
    N = 3 * L**3 # Total number of spins
    M = 4 * L**3 # Total number of interactions

    # Initialize the X and Z matrices and the sign vector
    X = np.zeros((M, N), dtype=int)
    Z = np.zeros((M, N), dtype=int)
    s = np.zeros(M, dtype=int)  # sign vector (0 for +, 1 for -)
    
    # Counter to keep track of the number of operators added
    counter = 0

    # Build the four check operators for all the red cubes
    for l in range(0, 2 * L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
        for r in range(0, 2 * L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
            for c in range(L): 

                if ((l // 2 + r // 2 + c) % 2) == 0: # Condition for the cube to be red

                    # Represent each of the four check operators for this cube in one row of the tableau:
                    # Down bottom right corner
                    X[counter, onedpostoric(L, c, r, l)] = 1 #site
                    X[counter, onedpostoric(L, (c + 1) % L, r + 1, l)] = 1 #north
                    X[counter, onedpostoric(L, (c + 1) % L, r // 2, l + 1)] = 1 #up
                    counter += 1

                    # Down top left corner
                    X[counter, onedpostoric(L, c, r + 1, l)] = 1 #south
                    X[counter, onedpostoric(L, c, (r + 2) % (2 * L), l)] = 1 #east
                    X[counter, onedpostoric(L, c, (r // 2 + 1)%L, l + 1)] = 1 #up
                    counter += 1

                    # Up bottom left corner
                    X[counter, onedpostoric(L, c, r // 2, l + 1)] = 1 #down
                    X[counter, onedpostoric(L, c, r, (l + 2) % (2 * L))] = 1 #east
                    X[counter, onedpostoric(L, c, r + 1, (l + 2) % (2 * L))] = 1 #north
                    counter += 1

                    # Up top right corner
                    X[counter, onedpostoric(L, (c + 1) % L, (r // 2 + 1) % L, l + 1)] = 1 #down
                    X[counter, onedpostoric(L, c, (r + 2) % (2 * L), (l + 2) % (2 * L))] = 1 #west
                    X[counter, onedpostoric(L, (c + 1) % L, r + 1, (l + 2) % (2 * L))] = 1 #south
                    counter += 1
    
    # Build the four check operators for all the blue cubes
    for l in range(0, 2 * L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
        for r in range(0, 2 * L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
            for c in range(L): 

                if ((l // 2 + r // 2 + c) % 2) != 0: # Condition for the cube to be blue

                    # Represent each of the four check operators for this cube in one row of the tableau:
                    # Down bottom left corner
                    Z[counter, onedpostoric(L, c, r, l)] = 1 #site
                    Z[counter, onedpostoric(L, c, r + 1, l)] = 1 #north
                    Z[counter, onedpostoric(L, c, r // 2, l + 1)] = 1 #up
                    counter += 1

                    # Down top right corner
                    Z[counter, onedpostoric(L, (c + 1) % L, r + 1, l)] = 1 #south
                    Z[counter, onedpostoric(L, c, (r + 2) % (2 * L), l)] = 1 #west
                    Z[counter, onedpostoric(L, (c + 1) % L, (r // 2 + 1) % L, l + 1)] = 1 #up
                    counter += 1

                    # Up bottom right corner
                    Z[counter, onedpostoric(L, (c + 1) % L, r // 2, l + 1)] = 1 #down
                    Z[counter, onedpostoric(L, c, r, (l + 2) % (2 * L))] = 1 #west
                    Z[counter, onedpostoric(L, (c + 1) % L, r + 1, (l + 2) % (2 * L))] = 1 #north
                    counter += 1

                    # Up top left corner
                    Z[counter, onedpostoric(L, c, (r // 2 + 1) % L, l + 1)] = 1 #down
                    Z[counter, onedpostoric(L, c, r + 1, (l + 2) % (2 * L))] = 1 #south
                    Z[counter, onedpostoric(L, c, (r + 2) % (2 * L), (l + 2) % (2 * L))] = 1 #east
                    counter += 1
                          
    return X, Z, s


def stabilizer_subsystem_toric_matrices(L):
    """
    Constructs binary matrices representing a simplified version of the subsystem toric code Hamiltonian; consisting
    only of commuting cube interactions. The model is defined on a 3D square lattice with periodic boundary
    conditions. There is one spin per edge of the lattice

    Input: Integer L which specifies the dimensions of the lattice (in number of cubes)
 
    Returns: X,Z,s 

    The qubits are labeled in a bottom-up fashion. For each layer, the spins are numbered also in
    a bottom-up fashion (see toric_3D_matrices)
    """

    # Total number of spins
    N = 3*L**3
    # Total number of interactions
    M = L**3

    # Initialize the X and Z matrices and the sign vector
    X = np.zeros((M, N), dtype=int)
    Z = np.zeros((M, N), dtype=int)
    s = np.zeros(M, dtype=int)  # sign vector (0 for +, 1 for -)
    
    # Counter to keep track of the number of operators added
    counter = 0

    # Build the cube operators A_b for all red cubes (see also corresponding figure in the accompanying paper)
    for l in range(0,2*L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
        for r in range(0, 2*L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
            for c in range(L): 
                
                if ((l//2 + r//2 + c) % 2) == 0: # Condition for the cube to be red

                    # Represent the stabilizer operator for this cube in one row of the tableau:
                    # Bottom face
                    X[counter, onedpostoric(L, c, r, l)] = 1 #site
                    X[counter, onedpostoric(L, c, r + 1, l)] = 1 #west
                    X[counter, onedpostoric(L, c, (r + 2) % (2 * L), l)] = 1 #north
                    X[counter, onedpostoric(L, (c + 1) % L, r + 1, l)] = 1 #east

                    # Top face
                    X[counter, onedpostoric(L, c, r, (l + 2) % (2 * L))] = 1 #south
                    X[counter, onedpostoric(L, c, r + 1, (l + 2) % (2 * L))] = 1 #west
                    X[counter, onedpostoric(L, (c + 1) % L, r + 1, (l + 2) % (2 * L))] = 1 #east
                    X[counter, onedpostoric(L, c, (r + 2) % (2 * L), (l + 2) % (2 * L))] = 1 #north

                    # Layer corresponding to the edges in-between the bottom and top faces
                    X[counter, onedpostoric(L, c, r // 2, l + 1)] = 1 #south west
                    X[counter, onedpostoric(L, (c + 1) % L, r // 2, l + 1)] = 1 #south east
                    X[counter, onedpostoric(L, c, (r // 2 + 1) % L, l + 1)] = 1 #north west
                    X[counter, onedpostoric(L, (c + 1) % L, (r // 2 + 1) % L, l + 1)] = 1 #north east
                    counter += 1

    # Build the cube operators B_p for all blue cubes
    for l in range(0,2*L,2): # Iterate only over the spins situated in the edges contained in the xy-plane
        for r in range(0, 2*L, 2): # Iterate only over the spins situated in the edges contained in the y-axis
            for c in range(L): 
                
                if ((l//2 + r//2 + c) % 2) != 0: # Condition for the cube to be blue

                    # Represent the stabilizer operator for this cube in one row of the tableau:
                    # Bottom face
                    Z[counter, onedpostoric(L, c, r, l)] = 1 #site
                    Z[counter, onedpostoric(L, c, r + 1, l)] = 1 #west
                    Z[counter, onedpostoric(L, c, (r + 2) % (2 * L), l)] = 1 #north
                    Z[counter, onedpostoric(L, (c + 1) % L, r + 1, l)] = 1 #east

                    # Top face
                    Z[counter, onedpostoric(L, c, r, (l + 2) % (2 * L))] = 1 #south
                    Z[counter, onedpostoric(L, c, r + 1, (l + 2) % (2 * L))] = 1 #west
                    Z[counter, onedpostoric(L, (c + 1) % L, r + 1, (l + 2) % (2 * L))] = 1 #east
                    Z[counter, onedpostoric(L, c, (r + 2) % (2 * L), (l + 2) % (2 * L))] = 1 #north

                    # Layer corresponding to the edges in-between the bottom and top faces
                    Z[counter, onedpostoric(L, c, r // 2, l + 1)] = 1 #south west
                    Z[counter, onedpostoric(L, (c + 1) % L, r // 2, l + 1)] = 1 #south east
                    Z[counter, onedpostoric(L, c, (r // 2 + 1) % L, l + 1)] = 1 #north west
                    Z[counter, onedpostoric(L, (c + 1) % L, (r // 2 + 1) % L, l + 1)] = 1 #north east
                    counter += 1
    
    return X, Z, s


def rotated_surface_matrices(L):
    """
    Constructs binary matrices representing the rotated surface code Hamiltonian on a square lattice with open
    boundary conditions. There is one spin situated at each vertex of the lattice.

    Input: Integer L which specifies the dimensions of the lattice (in number of cubes)
    
    Returns: X,Z,s
    
    The qubits are labeled in a bottom-up fashion. For each layer, the spins are numbered also in
    a bottom-up fashion

    .
    .
    .

    L+1    L+2     L+3       ....

    0       1       2       ....
    """
    # Total number of spins
    N = (L+1)**2 
    # Total number of interactions
    M = L**2 + 2*L

    # Initialize the X and Z matrices and the sign vector
    X = np.zeros((M, N), dtype=int)
    Z = np.zeros((M, N), dtype=int)
    s = np.zeros(M, dtype=int)  # sign vector (0 for +, 1 for -)
    
    # Counter to keep track of the number of operators added
    counter = 0

    # Generate the four-spin (square) operators (see also corresponding figure in the accompanying paper)
    for y in range(L): 
        for x in range(L):
            if (x + y) % 2 == 0: # Condition for the square to be red

                # Represent the A_p operator in one row of the tableau:
                X[counter, onedpossurface(L, x, y)] = 1 #bottom left
                X[counter, onedpossurface(L, x, y+1)] = 1 #top left left
                X[counter, onedpossurface(L, x+1, y)] = 1 #bottom right
                X[counter, onedpossurface(L, x+1, y+1)] = 1 #top right
                counter += 1

            else: # Otherwise the square is colored blue

                # Represent the B_p operator in one row of the tableau:
                Z[counter, onedpossurface(L, x, y)] = 1 #bottom left
                Z[counter, onedpossurface(L, x, y+1)] = 1 #top left left
                Z[counter, onedpossurface(L, x+1, y)] = 1 #bottom right
                Z[counter, onedpossurface(L, x+1, y+1)] = 1 #top right
                counter += 1
    
    # Generate semicircle operators at the vertical boundaries of the lattice 
    for y in range(L): 
        if y % 2 == 1:
            X[counter, onedpossurface(L, 0, y)] = 1 #down
            X[counter, onedpossurface(L, 0, y+1)] = 1 #up
            counter += 1
        
        if (y + L - 1) % 2 == 1:
            X[counter, onedpossurface(L, L, y)] = 1 #down
            X[counter, onedpossurface(L, L, y+1)] = 1 #up
            counter += 1
    
    # Generate semicircle operators at the horizontal boundaries of the lattice
    for x in range(L):
        if x % 2 == 0: 
            Z[counter, onedpossurface(L, x, 0)] = 1 #left
            Z[counter, onedpossurface(L, x+1, 0)] = 1 #right
            counter += 1
        if (x + L -1) % 2 == 0: 
            Z[counter, onedpossurface(L, x, L)] = 1 #left
            Z[counter, onedpossurface(L, x+1, L)] = 1 #right
            counter += 1
    
    return X, Z, s


# -------------------------------
# Main Routines
# -------------------------------
"""
In model_config we specify for each model:
- "matrix_func": the function that returns (X, Z, s) for that specific model
- "single_size": the lattice size to use in a single run
- "checker_sizes": a function (or iterable) that yields sizes for the checker
- "size_constraints": an optional function that filters out invalid sizes in checker
"""
model_config = {
    "toric_code": {
        "matrix_func": toric_code_matrices,
        "single_size": 3,
        "checker_sizes": lambda: range(2, 10),
        "size_constraints": None,
    },
    "color_honeycomb": {
        "matrix_func": color_honeycomb_matrices,
        "single_size": 2,
        "checker_sizes": lambda: range(2, 5),
        "size_constraints": None,
    },
    "haahs_code": {
        "matrix_func": haah_matrices,
        "single_size": 5,
        "checker_sizes": lambda: range(3, 7, 2),
        # Exclude L if it is even or a multiple of 4^p - 1 for some p≥2. We only exclude the first three numbers of such kind for simplicity 
        # So we want to skip 15, 63 and 255 in the checker:
        "size_constraints": lambda L: L != 15 and L != 63 and L != 255,
    },
    "3D_toric": {
        "matrix_func": toric_3D_matrices,
        "single_size": 3,
        "checker_sizes": lambda: range(2, 21),
        "size_constraints": None,
    },
    "X_cube": {
        "matrix_func": X_cube_matrices,
        "single_size": 3,
        "checker_sizes": lambda: range(2, 5),
        "size_constraints": None,
    },
    "checks_subsystem_toric": {
        "matrix_func": checks_subsystem_toric_matrices,
        "single_size": 2,
        "checker_sizes": lambda: range(2, 5, 2),  # Need even L
        "size_constraints": lambda L: L % 2 == 0,
    },
    "stabilizer_subsystem_toric": {
        "matrix_func": stabilizer_subsystem_toric_matrices,
        "single_size": 2,
        "checker_sizes": lambda: range(2, 5, 2), # Need even L
        "size_constraints": lambda L: L % 2 == 0,
    },
    "rotated_surface": {
        "matrix_func": rotated_surface_matrices,
        "single_size": 3,
        "checker_sizes": lambda: range(2, 5),
        "size_constraints": None,
    },
}


def run_single(model_key):
    """
    Perform exactly one diagonalization + Clifford‐gate application + Z‐simplification
    for the chosen model at its designated "single_size" lattice size (see model_config).
    """
    # Extract important parameters from model_config
    cfg = model_config[model_key]
    func = cfg["matrix_func"]
    size = cfg["single_size"]
    constraint = cfg.get("size_constraints")


    if constraint and not constraint(size):
        print("The model cannot be run for the current size")
        return
        
    # Generate the matrices and the tableaus for the model at the specified size
    X, Z, s = func(size)
    initial_tableau = Tableau(X, Z, s)
    diag_tableau = copy.deepcopy(initial_tableau)
    physical_tableau = copy.deepcopy(initial_tableau)

    # Print the initial tableau
    print(f"=== Single run for {model_key} model (size={str(size)}) ===")
    print("\nInitial tableau:")
    initial_tableau.print_tableaus_X_Z()

    # Perform full diagonalization on initial_tableau
    rank = diag_tableau.full_diagonalization()
    # Run only Clifford gates on the physical_tableau copy
    physical_tableau.apply_only_clifford_gates(diag_tableau.gate_log)
    
    #If the model checked corresponds to the 3D toric code, check the locality of the model
    if model_key == "3D_toric": 
        mr, mc = check_matrix_sums(physical_tableau.Z)
        log_message("Max row sum, max col sum: " + str(mr) + ", " + str(mc))
    
    # Use pseudo gaussian elimination to simplify the Z tableau of physical_tableau further
    # (X tableau should already be fully cleared)
    physical_tableau.simplify_Z()

    # Print the final tableau, its corresponding interactions and boolean, indicating whether the final tableau
    # has the correct predicted form
    print("\nFinal tableau:")
    physical_tableau.print_tableaus_X_Z()
    physical_tableau.print_final_interactions()
    print(f"\nDoes the final tabelau have the predicted form?: {physical_tableau.check_final_Z_tableau(model_key)}")
    print("\n")


def run_checker(model_key):
    """
    Loop over all designated sizes for a given model, performing diagonalization,
    Clifford‐gate application, and Z‐simplification.
    Logs errors if the final check, whether the tableau has the correct predicted form, gives False.
    Stores results in 'output.txt' and prints them to the console.
    """
    # Extract important parameters from model_config
    cfg = model_config[model_key]
    func = cfg["matrix_func"]
    sizes_iter = cfg["checker_sizes"]()
    constraint = cfg.get("size_constraints")

    # Iterate over all allowed sizes of the model (from "checker_sizes" in model_config)
    for size in sizes_iter:
        
        # If a size constraint is defined, check that the current size parameter satisfies it (otherwise skip)
        if constraint and not constraint(size):
            continue

        # Generate the matrices and the tableaus for the model at this size
        log_message(f"[checker] {model_key}: working with size={str(size)}")
        X, Z, s = func(size)
        initial_tableau = Tableau(X, Z, s)
        diag_tableau = copy.deepcopy(initial_tableau)
        physical_tableau = copy.deepcopy(initial_tableau)

        # Perform full diagonalization on initial_tableau
        rank = diag_tableau.full_diagonalization()
        # Run only Clifford gates on the physical_tableau copy
        physical_tableau.apply_only_clifford_gates(diag_tableau.gate_log)

        #If the model checked corresponds to the 3D toric code, check the locality of the model
        if model_key == "3D_toric": 
            mr, mc = check_matrix_sums(physical_tableau.Z)
            log_message("Max row sum, max col sum: " + str(mr) + ", " + str(mc))

        # Use pseudo gaussian elimination to simplify the Z tableau of physical_tableau further
        # (X tableau should already be fully cleared)
        physical_tableau.simplify_Z()

        # Log an error message if the final tableau does not have the predicted form
        if not physical_tableau.check_final_Z_tableau(model_key):
            log_message(f"ERROR: {model_key} failed at size={str(size)}")


def main():
    usage = (
        "\nUse one of the following two commands:\n"
        "  python path/program_name.py single_run <model_name>\n"
        "  python path/program_name.py checker <model_name>\n\n"
        "If you omit <model_name> for 'checker', it will run over all models.\n"
        "Available models: " + ", ".join(model_config.keys()) + "\n\n"
        "Checker will save the results in 'output.txt' and print them in the console.\n"
        )
    
    # The command must have two or three arguments
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Error: Invalid number of arguments {len(sys.argv)}.\n")
        print(usage)
        return

    # Second argument must be either 'single_run' or 'checker'
    command = sys.argv[1]

    # Handle 'single_run'
    if command == "single_run":
        # single_run requires exactly one model name
        if len(sys.argv) != 3:
            print("Error: 'single_run' requires exactly one <model_name>.\n")
            print(usage)
            return

        # Check if the model name is valid
        model_name = sys.argv[2]
        if model_name not in model_config:
            print(f"Unknown model: '{model_name}'\n")
            print("Available models:", ", ".join(model_config.keys()))
            return

        # Run the single run for the specified model
        run_single(model_name)

    # Handle 'checker' (model name is optional here)
    elif command == "checker":

        # Check if name of a model is provided
        if len(sys.argv) == 2:
            # If no model specified, run over all models
            print("No model specified, running checker on all models...\n")
            for model_key in model_config:
                print(f"\n=== Running checker on '{model_key}' ===")
                run_checker(model_key)

        elif len(sys.argv) == 3:
            model_key = sys.argv[2]
            # If a model is specified, check if it is valid
            if model_key not in model_config:
                print(f"Unknown model: '{model_key}'\n")
                print("Available models:", ", ".join(model_config.keys()))
                return
            # Run the checker for the specified model
            run_checker(model_key)

    # If the command is neither 'single_run' nor 'checker', print an error
    else:
        print(f"Unknown command: '{command}'.\n")
        print(usage)

if __name__ == "__main__":

    main()
