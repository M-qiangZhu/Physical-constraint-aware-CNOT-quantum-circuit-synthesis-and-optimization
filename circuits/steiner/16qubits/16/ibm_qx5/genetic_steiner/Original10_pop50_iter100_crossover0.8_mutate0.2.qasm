// Initial wiring: [6, 4, 2, 10, 11, 5, 14, 3, 7, 8, 13, 9, 0, 1, 12, 15]
// Resulting wiring: [6, 4, 2, 10, 11, 5, 14, 3, 7, 8, 13, 9, 0, 1, 12, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[4], q[3];
cx q[3], q[2];
cx q[8], q[7];
cx q[10], q[5];
cx q[5], q[4];
cx q[12], q[11];
cx q[13], q[12];
cx q[12], q[11];
cx q[13], q[2];
cx q[13], q[12];
cx q[10], q[11];
cx q[9], q[10];
cx q[7], q[8];
cx q[1], q[14];
cx q[14], q[15];
cx q[0], q[15];
