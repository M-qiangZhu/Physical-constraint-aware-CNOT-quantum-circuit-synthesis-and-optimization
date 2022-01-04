// Initial wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// Resulting wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[14], q[7];
cx q[15], q[12];
cx q[3], q[6];
cx q[1], q[10];
cx q[0], q[1];
cx q[12], q[2];
cx q[4], q[11];
cx q[9], q[2];
