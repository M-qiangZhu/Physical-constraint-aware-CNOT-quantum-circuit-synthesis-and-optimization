// Initial wiring: [3, 1, 8, 2, 5, 6, 0, 7, 4]
// Resulting wiring: [3, 1, 8, 2, 5, 6, 0, 7, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[0], q[5];
cx q[0], q[1];
