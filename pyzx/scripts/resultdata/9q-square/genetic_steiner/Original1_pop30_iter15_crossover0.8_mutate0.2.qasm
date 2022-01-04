// Initial wiring: [0, 2, 3, 7, 8, 1, 4, 5, 6]
// Resulting wiring: [0, 2, 3, 7, 8, 1, 4, 5, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[4];
cx q[3], q[4];
cx q[1], q[4];
