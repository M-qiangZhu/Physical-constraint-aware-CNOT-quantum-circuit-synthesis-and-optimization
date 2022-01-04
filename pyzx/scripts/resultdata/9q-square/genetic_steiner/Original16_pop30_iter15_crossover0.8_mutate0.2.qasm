// Initial wiring: [2, 0, 8, 7, 4, 1, 6, 5, 3]
// Resulting wiring: [2, 0, 8, 7, 4, 1, 6, 5, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[7], q[8];
cx q[0], q[5];
