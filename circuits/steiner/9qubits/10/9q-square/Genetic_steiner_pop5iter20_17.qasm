// Initial wiring: [4, 2, 3, 7, 1, 0, 6, 8, 5]
// Resulting wiring: [4, 2, 3, 7, 1, 0, 6, 8, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[6], q[7];
cx q[7], q[8];
cx q[3], q[8];
cx q[6], q[7];
cx q[2], q[3];
cx q[1], q[0];
cx q[5], q[0];
cx q[4], q[1];
