// Initial wiring: [1, 2, 5, 6, 3, 4, 0, 8, 7]
// Resulting wiring: [1, 2, 5, 6, 3, 4, 0, 8, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[8];
cx q[7], q[6];
cx q[6], q[5];
cx q[4], q[1];
cx q[5], q[0];
cx q[6], q[5];
