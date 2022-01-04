// Initial wiring: [2, 8, 7, 0, 4, 1, 6, 5, 3]
// Resulting wiring: [2, 8, 7, 0, 4, 1, 6, 5, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[4];
cx q[1], q[2];
cx q[0], q[5];
