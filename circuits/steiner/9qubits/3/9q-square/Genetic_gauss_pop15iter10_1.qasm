// Initial wiring: [0 1 3 4 2 5 6 7 8]
// Resulting wiring: [0 1 3 4 2 5 6 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[2], q[1];
cx q[0], q[1];
