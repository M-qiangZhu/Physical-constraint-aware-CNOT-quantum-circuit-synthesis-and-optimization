// Initial wiring: [8, 7, 1, 3, 4, 5, 0, 2, 6]
// Resulting wiring: [8, 7, 1, 3, 4, 5, 0, 2, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[5];
cx q[4], q[3];
cx q[3], q[2];
