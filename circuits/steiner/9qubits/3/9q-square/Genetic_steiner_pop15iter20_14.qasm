// Initial wiring: [2, 8, 7, 4, 5, 6, 0, 1, 3]
// Resulting wiring: [2, 8, 7, 4, 5, 6, 0, 1, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[5], q[6];
cx q[3], q[8];
