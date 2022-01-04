// Initial wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// Resulting wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[4];
cx q[2], q[14];
cx q[7], q[11];
cx q[7], q[6];
cx q[0], q[14];
cx q[11], q[12];
cx q[11], q[0];
cx q[14], q[5];
