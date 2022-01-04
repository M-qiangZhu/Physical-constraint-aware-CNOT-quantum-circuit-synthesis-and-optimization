// Initial wiring: [5, 8, 13, 14, 3, 9, 0, 1, 11, 4, 7, 2, 6, 10, 12, 15]
// Resulting wiring: [5, 8, 13, 14, 3, 9, 0, 1, 11, 4, 7, 2, 6, 10, 12, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[6], q[5];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[7], q[0];
cx q[9], q[6];
cx q[13], q[12];
cx q[12], q[11];
cx q[13], q[12];
cx q[14], q[9];
cx q[9], q[6];
cx q[14], q[9];
cx q[14], q[15];
cx q[10], q[11];
cx q[9], q[10];
cx q[8], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[6], q[9];
cx q[3], q[4];
cx q[2], q[5];
