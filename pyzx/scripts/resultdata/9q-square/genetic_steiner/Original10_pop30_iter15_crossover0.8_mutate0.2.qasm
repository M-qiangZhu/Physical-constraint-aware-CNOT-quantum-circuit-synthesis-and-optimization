// Initial wiring: [0, 6, 4, 8, 1, 5, 2, 3, 7]
// Resulting wiring: [0, 6, 4, 8, 1, 5, 2, 3, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[3];
cx q[5], q[4];
cx q[1], q[4];
