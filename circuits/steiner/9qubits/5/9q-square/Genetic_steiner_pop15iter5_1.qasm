// Initial wiring: [0, 4, 3, 2, 1, 6, 8, 7, 5]
// Resulting wiring: [0, 4, 3, 2, 1, 6, 8, 7, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[0], q[5];
cx q[5], q[6];
cx q[0], q[5];
cx q[3], q[8];
cx q[2], q[3];
