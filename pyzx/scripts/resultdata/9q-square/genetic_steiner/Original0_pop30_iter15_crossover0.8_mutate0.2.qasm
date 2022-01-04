// Initial wiring: [5, 4, 2, 8, 0, 6, 3, 1, 7]
// Resulting wiring: [5, 4, 2, 8, 0, 6, 3, 1, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[1];
cx q[5], q[0];
cx q[8], q[7];
