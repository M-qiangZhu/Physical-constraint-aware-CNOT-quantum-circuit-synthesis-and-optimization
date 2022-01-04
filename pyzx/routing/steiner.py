# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2019 - Aleks Kissinger, John van de Wetering,
#                      and Arianne Meijer-van de Griend

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from . import architecture
from ..linalg import Mat2

debug = False

def steiner_gauss(matrix, architecture, full_reduce=False, x=None, y=None, permutation=None):
    """
    Performs Gaussian elimination that is constraint bij the given architecture    
    :param matrix: PyZX Mat2 matrix to be reduced
    :param architecture: The Architecture object to conform to
    :param full_reduce: Whether to fully reduce or only create an upper triangular form  
    :param x: 
    :param y: 
    :return: Rank of the given matrix  
    """
    debug and print("steiner_guass is called.")
    debug and print(matrix)
    debug and print("architecture:",architecture.name)
    debug and print("full_reduce:",full_reduce)
    debug and print("x:",x)
    debug and print("y:",y.matrix)  
    debug and print("permutation:",permutation)
    
    if permutation is None:
        permutation = [i for i in range(len(matrix.data))]
        debug and print("permutation:",permutation)
    else:
        matrix = Mat2([matrix.data[i] for i in permutation])
    # print(matrix)
    # print("*" * 50)
    
    def row_add(c0, c1):
        debug and print("-" * 60)
        matrix.row_add(c0, c1) 
        
        c0 = permutation[c0]
        debug and print("起始行c0：",c0)
        
        c1 = permutation[c1]
        debug and print("目标行c1：",c1)
        
        debug and print("Reducing", c0, c1)
        
        debug and print("x[运算前]：",x)
        if x != None: x.row_add(c0, c1)
        debug and print("x[运算后]：",x)
        
        debug and print("y[运算前]：",y.matrix)
        if y != None: y.col_add(c1, c0)
        debug and print("y[运算后]：\n",y.matrix)
        
        
    
    def steiner_reduce(col, root, nodes, upper):

        
        steiner_tree = architecture.steiner_tree(root, nodes, upper)
        # steiner_tree = architecture.my_steiner_tree(root, nodes, upper)
        # print(f"steiner_tree :")
        # for i in range(0, len(steiner_tree) - 1):
        #     print(i)
        
        # Remove all zeros 
        next_check = next(steiner_tree)  
        debug and print("next_check:",next_check)
        
        debug and print("<-----------------------Step 1: remove zeros-------------------------->")
        if upper:
            zeros = []
            while next_check is not None:
                s0, s1 = next_check
                if matrix.data[s0][col] == 0:  
                    zeros.append(next_check)  
                next_check = next(steiner_tree)
                debug and print("next_check again:",next_check)
            while len(zeros) > 0:
                s0, s1 = zeros.pop(-1)
                if matrix.data[s0][col] == 0:

                    row_add(s1, s0)
                    cnts += 1
                    debug and print("传递1后的当前矩阵为:")
                    debug and print(matrix.data)
                    debug and print(f"a[{s0}][{col}]={matrix.data[s0][col]},\na[{s1}][{col}]={matrix.data[s1][col]}")
        
        else:
            debug and print("deal with zero root")
            if next_check is not None and matrix.data[next_check[0]][col] == 0:  # root is zero
                print("WARNING : Root is 0 => reducing non-pivot column", matrix.data)
            debug and print("Step 1: remove zeros", [r[c] for r in matrix.data])
            while next_check is not None:
                s0, s1 = next_check
                if matrix.data[s1][col] == 0:  # s1 is a new steiner point
                    row_add(s0, s1)
                    cnts += 1
                    debug and print("matrix after remove zeros:")
                    debug and print(f"a[{s0}][{col}]={matrix.data[s0][col]},\na[{s1}][{col}]={matrix.data[s1][col]}")
                    # debug and print(matrix.data)
                next_check = next(steiner_tree)

        
        # Reduce stuff
        debug and print("<<----------------------Step 2: remove ones-------------------------->>")
        next_add = next(steiner_tree)
        while next_add is not None:
            s0, s1 = next_add
            row_add(s0, s1)
            cnts += 1
            debug and print("matrix after remove ones:")
            debug and print(matrix.data)
            next_add = next(steiner_tree)
            debug and print(next_add)
        debug and print("<<<--------------------Step 3: -------------------------->>>")
        return cnts



    rows = matrix.rows()
    debug and print("rows:",rows)
    cols = matrix.cols()
    debug and print("cols:",cols)
    p_cols = []
    pivot = 0


    for c in range(cols):
        debug and print(f"下三角第{c}列:")
        if pivot < rows:
            nodes = [r for r in range(pivot, rows) if pivot==r or matrix.data[r][c] == 1]  #
            debug and print("nodes:",nodes)
            cnot_cnts += steiner_reduce(c, pivot, nodes, True)  
            if matrix.data[pivot][c] == 1:
                p_cols.append(c)
                pivot += 1
                
    debug and print("Upper triangle form: ")
    debug and print(matrix.data)

    
    rank = pivot
    debug and print(f"p_cols: {p_cols}")
    
    if full_reduce:
        pivot -= 1
        for c in reversed(p_cols):
            debug and print(f"上三角第{c}列:")
            debug and print(pivot, [r[c] for r in matrix.data])
            
            nodes = [r for r in range(0, pivot+1) if r==pivot or matrix.data[r][c] == 1]
            if len(nodes) > 1:
                cnot_cnts += steiner_reduce(c, pivot, nodes, False)
            
            pivot -= 1

            
    return rank
