// Initial wiring: [8, 9, 6, 10, 15, 0, 7, 5, 2, 3, 14, 13, 12, 11, 1, 4]
// Resulting wiring: [8, 9, 6, 10, 15, 0, 7, 5, 2, 3, 14, 13, 12, 11, 1, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[10], q[5];
cx q[11], q[10];
cx q[10], q[5];
cx q[11], q[4];
cx q[12], q[11];
cx q[11], q[4];
cx q[13], q[12];
cx q[12], q[11];
cx q[11], q[4];
cx q[12], q[11];
cx q[14], q[9];
cx q[15], q[14];
cx q[14], q[9];
cx q[15], q[8];
cx q[15], q[14];
cx q[13], q[14];
cx q[9], q[10];
cx q[7], q[8];
cx q[8], q[9];
cx q[3], q[4];
cx q[4], q[5];
