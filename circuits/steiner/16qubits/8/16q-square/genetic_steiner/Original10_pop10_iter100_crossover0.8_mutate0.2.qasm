// Initial wiring: [6, 15, 3, 0, 7, 11, 4, 5, 1, 14, 10, 2, 12, 9, 8, 13]
// Resulting wiring: [6, 15, 3, 0, 7, 11, 4, 5, 1, 14, 10, 2, 12, 9, 8, 13]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[3], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[3], q[2];
cx q[4], q[3];
cx q[7], q[0];
cx q[9], q[6];
cx q[12], q[11];
cx q[5], q[6];
cx q[2], q[3];
cx q[0], q[1];
