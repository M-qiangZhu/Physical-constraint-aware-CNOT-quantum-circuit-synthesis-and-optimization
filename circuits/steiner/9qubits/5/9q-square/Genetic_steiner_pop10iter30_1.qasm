// Initial wiring: [0, 8, 6, 7, 2, 3, 4, 1, 5]
// Resulting wiring: [0, 8, 6, 7, 2, 3, 4, 1, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[7], q[8];
cx q[4], q[7];
cx q[7], q[8];
cx q[5], q[4];
cx q[2], q[1];
