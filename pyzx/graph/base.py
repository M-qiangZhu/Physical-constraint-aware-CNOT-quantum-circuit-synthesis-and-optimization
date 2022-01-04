# PyZX - Python library for quantum circuit rewriting 
#        and optimisation using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

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

import abc

from pyzx.tensor import tensorfy

class DocstringMeta(abc.ABCMeta):
    """Metaclass that allows docstring 'inheritance'."""

    def __new__(mcls, classname, bases, cls_dict):
        cls = abc.ABCMeta.__new__(mcls, classname, bases, cls_dict)
        mro = cls.__mro__[1:]
        for name, member in cls_dict.items():
            if not getattr(member, '__doc__'):
                for base in mro:
                    try:
                        member.__doc__ = getattr(base, name).__doc__
                        break
                    except AttributeError:
                        pass
        return cls

def pack_indices(lst):
    d = dict()
    if len(lst) == 0: return d
    list.sort(lst)
    i = 0
    x = None
    for j in range(len(lst)):
        y = lst[j]
        if y != x:
            x = y
            d[y] = i
            i += 1
    return d

class BaseGraph(object):
    """Base class for letting graph backends interact with PyZX.
    For a backend to work with PyZX, there should be a class that implements
    all the methods of this class. For implementations of this class see 
    :class:`~graph.graph_s.GraphS` or :class `~graph.graph_ig.GraphIG`."""
    __metaclass__ = DocstringMeta
    backend = 'None'

    def __init__(self):
        self.track_phases = False
        self.phase_index = dict()
        self.phase_master = None
        self.phase_mult = dict()
        self.max_phase_index = -1

    def __str__(self):
        return "Graph({} vertices, {} edges)".format(
                str(self.num_vertices()),str(self.num_edges()))

    def __repr__(self):
        return str(self)

    def stats(self):
        s = str(self) + "\n"
        degrees = {}
        for v in self.vertices():
            d = self.vertex_degree(v)
            if d in degrees: degrees[d] += 1
            else: degrees[d] = 1
        s += "degree distribution: \n"
        for d, n in sorted(degrees.items(),key=lambda x: x[0]):
            s += "{:d}: {:d}\n".format(d,n)
        return s

    def copy(self, adjoint=False, backend=None):
        """Create a copy of the graph. If ``adjoint`` is set, 
        the adjoint of the graph will be returned (inputs and outputs flipped, phases reversed).
        When ``backend`` is set, a copy of the graph with the given backend is produced. 
        By default the copy will have the same backend.

        `Note`: The copy will have consecutive vertex indices, even if the original
        graph did not.
        """
        from .graph import Graph # imported here to prevent circularity
        if (backend == None):
            backend = type(self).backend
        g = Graph(backend = backend)
        g.track_phases = self.track_phases
        mult = 1
        if adjoint: mult = -1

        #g.add_vertices(self.num_vertices())
        ty = self.types()
        ph = self.phases()
        qs = self.qubits()
        rs = self.rows()
        maxr = self.depth()
        vtab = dict()
        for v in self.vertices():
            i = g.add_vertex(ty[v],phase=mult*ph[v])
            if v in qs: g.set_qubit(i,qs[v])
            if v in rs: 
                if adjoint: g.set_row(i, maxr-rs[v])
                else: g.set_row(i, rs[v])
            vtab[v] = i
            for k in self.vdata_keys(v):
                g.set_vdata(i, k, self.vdata(v, k))

        for i in self.inputs:
            if adjoint: g.outputs.append(vtab[i])
            else: g.inputs.append(vtab[i])
        for o in self.outputs:
            if adjoint: g.inputs.append(vtab[o])
            else: g.outputs.append(vtab[o])
        
        etab = {e:(vtab[self.edge_s(e)],vtab[self.edge_t(e)]) for e in self.edges()}
        g.add_edges(etab.values())
        for e,(s,t) in etab.items():
            g.set_edge_type(g.edge(s,t), self.edge_type(e))
        return g
    def adjoint(self):
        """Returns a new graph equal to the adjoint of this graph."""
        return self.copy(adjoint=True)

    def replace_subgraph(self, left_row, right_row, replace):
        """Deletes the subgraph of all nodes with rank strictly between ``left_row``
        and ``right_row`` and replaces it with the graph ``replace``.
        The amount of nodes on the left row should match the amount of inputs of 
        the replacement graph and the same for the right row and the outputs.
        The graphs are glued together based on the qubit index of the vertices."""
        qleft = [v for v in self.vertices() if self.row(v)==left_row]
        qright= [v for v in self.vertices() if self.row(v)==right_row]
        if len(qleft) != len(replace.inputs):
            raise TypeError("Inputs do not match glueing vertices")
        if len(qright) != len(replace.outputs):
            raise TypeError("Outputs do not match glueing vertices")
        if set(self.qubit(v) for v in qleft) != set(replace.qubit(v) for v in replace.inputs):
            raise TypeError("Input qubit indices do not match")
        if set(self.qubit(v) for v in qright)!= set(replace.qubit(v) for v in replace.outputs):
            raise TypeError("Output qubit indices do not match")
        
        self.remove_vertices([v for v in self.vertices() if (left_row < self.row(v) and self.row(v) < right_row)])
        self.remove_edges([self.edge(s,t) for s in qleft for t in qright if self.connected(s,t)])
        rdepth = replace.depth() -1
        for v in (v for v in self.vertices() if self.row(v)>=right_row):
            self.set_row(v, self.row(v)+rdepth)

        vtab = {}
        for v in replace.vertices():
            if v in replace.inputs or v in replace.outputs: continue
            vtab[v] = self.add_vertex(replace.type(v),replace.qubit(v),
                                replace.row(v)+left_row,replace.phase(v))
        for v in replace.inputs:
            vtab[v] = [i for i in qleft if self.qubit(i) == replace.qubit(v)][0]

        for v in replace.outputs:
            vtab[v] = [i for i in qright if self.qubit(i) == replace.qubit(v)][0]

        etab = {e:(vtab[replace.edge_s(e)],vtab[replace.edge_t(e)]) for e in replace.edges()}
        self.add_edges(etab.values())
        for e,(s,t) in etab.items():
            self.set_edge_type(self.edge(s,t), replace.edge_type(e))

    def compose(self, other):
        """Inserts a circuit after this one. The amount of qubits of the circuits must match."""
        if self.qubit_count() != other.qubit_count():
            raise TypeError("Circuits work on different qubit amounts")
        self.normalise()
        other = other.copy()
        other.normalise()
        for o in self.outputs:
            q = self.qubit(o)
            e = list(self.incident_edges(o))[0]
            if self.edge_type(e) == 2: #hadamard edge
                i = [v for v in other.inputs if other.qubit(v)==q][0]
                e = list(other.incident_edges(i))[0]
                other.set_edge_type(e, 3-other.edge_type(e)) # toggle the edge type
        d = self.depth()
        self.replace_subgraph(d-1,d,other)

    def to_tensor(self):
        """Returns a representation of the graph as a tensor using :func:`~pyzx.tensor.tensorfy`"""
        return tensorfy(self)

    def vindex(self):
        """The index given to the next vertex added to the graph. It should always
        be equal to ``max(g.vertices()) + 1``."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def depth(self):
        """Returns the value of the highest row number given to a vertex.
        This is -1 when no rows have been set."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def pack_circuit_rows(self):
        """Compresses the rows of the graph so that every index is used."""
        rows = [self.row(v) for v in self.vertices()]
        new_rows = pack_indices(rows)
        for v in self.vertices():
            self.set_row(v, new_rows[self.row(v)])

    def qubit_count(self):
        """Returns the value of the highest qubit index given to a vertex.
        This is -1 when no qubit indices have been set."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def auto_detect_inputs(self):
        if self.inputs or self.outputs: return self.inputs, self.outputs
        minrow = 100000
        maxrow = -100000
        nodes = {}
        ty = self.types()
        for v in self.vertices():
            if ty[v] == 0:
                r = self.row(v)
                nodes[v] = r
                if r < minrow:
                    minrow = r
                if r > maxrow:
                    maxrow = r

        for v,r in nodes.items():
            if r == minrow:
                self.inputs.append(v)
            if r == maxrow:
                self.outputs.append(v)
        self.inputs.sort(key=self.qubit)
        self.outputs.sort(key=self.qubit)
        return self.inputs, self.outputs


    def normalise(self):
        """Puts every node connecting to an input/output at the correct qubit index and row."""
        if not self.inputs:
            self.auto_detect_inputs()
        max_r = self.depth() - 1
        if max_r <= 2: return
        claimed = []
        for q,i in enumerate(sorted(self.inputs, key=self.qubit)):
            self.set_row(i,0)
            self.set_qubit(i,q)
            #q = self.qubit(i)
            n = list(self.neighbours(i))[0]
            if self.type(n) in (1,2):
                claimed.append(n)
                self.set_row(n,1)
                self.set_qubit(n, q)
            else: #directly connected to output
                e = self.edge(i, n)
                t = self.edge_type(e)
                self.remove_edge(e)
                v = self.add_vertex(1,q,1)
                self.add_edge((i,v),3-t)
                self.add_edge((v,n), 2)
                claimed.append(v)
        for q, o in enumerate(sorted(self.outputs,key=self.qubit)):
            #q = self.qubit(o)
            self.set_row(o,max_r+1)
            self.set_qubit(o,q)
            n = list(self.neighbours(o))[0]
            if n not in claimed:
                self.set_row(n,max_r)
                self.set_qubit(n, q)
            else:
                e = self.edge(o, n)
                t = self.edge_type(e)
                self.remove_edge(e)
                v = self.add_vertex(1,q,max_r)
                self.add_edge((o,v),3-t)
                self.add_edge((v,n), 2)

        self.pack_circuit_rows()

    def add_vertices(self, amount):
        """Add the given amount of vertices, and return the indices of the
        new vertices added to the graph, namely: range(g.vindex() - amount, g.vindex())"""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def add_vertex(self, ty=0, qubit=-1, row=-1, phase=0):
        """Add a single vertex to the graph and return its index.
        The optional parameters allow you to respectively set
        the type, qubit index, row index and phase of the vertex."""
        v = self.add_vertices(1)[0]
        if ty: self.set_type(v, ty)
        if qubit!=-1: self.set_qubit(v, qubit)
        if row!=-1: self.set_row(v, row)
        if phase: 
            self.set_phase(v, phase)
        if self.track_phases:
            self.max_phase_index += 1
            self.phase_index[v] = self.max_phase_index
            self.phase_mult[v] = 1
        return v

    def add_edges(self, edges, edgetype=1):
        """Adds a list of edges to the graph. 
        If edgetype is 1 (the default), these will be regular edges.
        If edgetype is 2, these edges will be Hadamard edges."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def add_edge(self, edge, edgetype=1):
        """Adds a single edge of the given type (1=regular, 2=Hadamard edge)"""
        self.add_edges([edge], edgetype)

    def add_edge_table(self, etab):
        """Takes a dictionary mapping (source,target) --> (#edges, #h-edges) specifying that
        #edges regular edges must be added between source and target and $h-edges Hadamard edges.
        The method selectively adds or removes edges to produce that ZX diagram which would 
        result from adding (#edges, #h-edges), and then removing all parallel edges using Hopf/spider laws."""
        add = ([],[]) # list of edges and h-edges to add
        remove = []   # list of edges to remove
        #add_pi_phase = []
        for (v1,v2),(n1,n2) in etab.items():
            conn_type = self.edge_type(self.edge(v1,v2))
            if conn_type == 1: n1 += 1 #and add to the relevant edge count
            elif conn_type == 2: n2 += 1
            
            t1 = self.type(v1)
            t2 = self.type(v2)
            if t1 == t2:         #types are equal,
                n1 = bool(n1)     #so normal edges fuse
                n2 = n2%2         #while hadamard edges go modulo 2
                if n1 != 0 and n2 != 0:  #reduction rule for when both edges appear
                    new_type = 1
                    self.add_to_phase(v1, 1)
                    #add_pi_phase.append(v1)
                elif n1 != 0: new_type = 1
                elif n2 != 0: new_type = 2
                else: new_type = 0
            else:                #types are different
                n1 = n1%2        #so normal edges go modulo 2
                n2 = bool(n2)    #while hadamard edges fuse
                if n1 != 0 and n2 != 0:  #reduction rule for when both edges appear
                    new_type = 2
                    self.add_to_phase(v1, 1)
                    #add_pi_phase.append(v1)
                elif n1 != 0: new_type = 1
                elif n2 != 0: new_type = 2
                else: new_type = 0
            if new_type != 0: # They should be connected, so update the graph
                if conn_type == 0: #new edge added
                    add[new_type-1].append((v1,v2))
                elif conn_type != new_type: #type of edge has changed
                    self.set_edge_type(self.edge(v1,v2), new_type)
            elif conn_type != 0: #They were connected, but not anymore, so update the graph
                remove.append(self.edge(v1,v2))

        self.remove_edges(remove)
        self.add_edges(add[0],1)
        self.add_edges(add[1],2)

    def set_phase_master(self, m):
        self.phase_master = m

    def update_phase_index(self, old, new):
        if not self.track_phases: return
        i = self.phase_index[old]
        self.phase_index[old] = self.phase_index[new]
        self.phase_index[new] = i

    def fuse_phases(self, p1, p2):
        if p1 not in self.phase_index or p2 not in self.phase_index: 
            return
        if self.phase_master: 
            self.phase_master.fuse_phases(self.phase_index[p1],self.phase_index[p2])
        self.phase_index[p2] = self.phase_index[p1]

    def phase_negate(self, v):
        if v not in self.phase_index: return
        index = self.phase_index[v]
        #print("Negating phase", v, index)
        mult = self.phase_mult[index]
        self.phase_mult[index] = -1*mult 

    def vertex_from_phase_index(self, i):
        return list(self.phase_index.keys())[list(self.phase_index.values()).index(i)]


    def remove_vertices(self, vertices):
        """Removes the list of vertices from the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def remove_vertex(self, vertex):
        """Removes the given vertex from the graph."""
        self.remove_vertices([vertex])

    def remove_isolated_vertices(self):
        """Deletes all vertices that are not connected to any other vertex.
        Should be replaced by a faster alternative if available in the backend."""
        self.remove_vertices([v for v in self.vertices() if self.vertex_degree(v)==0])

    def remove_edges(self, edges):
        """Removes the list of edges from the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def remove_edge(self, edge):
        """Removes the given edge from the graph."""
        self.remove_edge([edge])

    def num_vertices(self):
        """Returns the amount of vertices in the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def num_edges(self):
        """Returns the amount of edges in the graph"""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def vertices(self):
        """Iterator over all the vertices."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def edges(self):
        """Iterator that returns all the edges. Output type depends on implementation in backend."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def vertex_set(self):
        """Returns the vertices of the graph as a Python set. 
        Should be overloaded if the backend supplies a cheaper version than this."""
        return set(self.vertices())

    def edge_set(self):
        """Returns the edges of the graph as a Python set. 
        Should be overloaded if the backend supplies a cheaper version than this."""
        return set(self.edges())

    def edge(self, s, t):
        """Returns the edge object with the given source/target."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def edge_st(self, edge):
        """Returns a tuple of source/target of the given edge."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def edge_s(self, edge):
        """Returns the source of the given edge."""
        return self.edge_st(edge)[0]
    def edge_t(self, edge):
        """Returns the target of the given edge."""
        return self.edge_st(edge)[1]

    def neighbours(self, vertex):
        """Returns all neighbouring vertices of the given vertex."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def vertex_degree(self, vertex):
        """Returns the degree of the given vertex."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def incident_edges(self, vertex):
        """Returns all neighbouring edges of the given vertex."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def connected(self,v1,v2):
        """Returns whether vertices v1 and v2 share an edge."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def edge_type(self, e):
        """Returns the type of the given edge:
        1 if it is regular, 2 if it is a Hadamard edge, 0 if the edge is not in the graph."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def set_edge_type(self, e, t):
        """Sets the type of the given edge."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)

    def type(self, vertex):
        """Returns the type of the given vertex:
        0 if it is a boundary, 1 if is a Z node, 2 if it a X node."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def types(self):
        """Returns a mapping of vertices to their types."""
        raise NotImplementedError("Not implemented on backend " + type(self).backend)
    def set_type(self, vertex, t):
        """Sets the type of the given vertex to t."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    
    def phase(self, vertex):
        """Returns the phase value of the given vertex."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def phases(self):
        """Returns a mapping of vertices to their phase values."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def set_phase(self, vertex, phase):
        """Sets the phase of the vertex to the given value."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def add_to_phase(self, vertex, phase):
        """Add the given phase to the phase value of the given vertex."""
        self.set_phase(vertex,self.phase(vertex)+phase)

    def qubit(self, vertex):
        """Returns the qubit index associated to the vertex. 
        If no index has been set, returns -1."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def qubits(self):
        """Returns a mapping of vertices to their qubit index."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def set_qubit(self, vertex, q):
        """Sets the qubit index associated to the vertex."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)

    def row(self, vertex):
        """Returns the row that the vertex is positioned at. 
        If no row has been set, returns -1."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def rows(self):
        """Returns a mapping of vertices to their row index."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def set_row(self, vertex, r):
        """Sets the row the vertex should be positioned at."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)

    def set_position(self, vertex, q, r):
        """Set both the qubit index and row index of the vertex."""
        self.set_qubit(vertex, q)
        self.set_row(vertex, r)

    def vdata_keys(self, vertex):
        """Returns an iterable of the vertex data key names.
        Used e.g. in making a copy of the graph in a backend-independent way."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def vdata(self, vertex, key, default=0):
        """Returns the data value of the given vertex associated to the key.
        If this key has no value associated with it, it returns the default value."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)
    def set_vdata(self, vertex, key, val):
        """Sets the vertex data associated to key to val."""
        raise NotImplementedError("Not implemented on backend" + type(self).backend)