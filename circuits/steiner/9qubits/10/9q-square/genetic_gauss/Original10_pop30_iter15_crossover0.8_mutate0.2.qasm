// Initial wiring: [2, 4, 5, 1, 6, 8, 3, 0, 7]
// Resulting wiring: [2, 4, 5, 1, 6, 8, 3, 0, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[1];
cx q[8], q[7];
cx q[6], q[0];
cx q[5], q[7];
cx q[4], q[8];
cx q[4], q[5];
cx q[3], q[7];
cx q[3], q[4];
cx q[2], q[7];
