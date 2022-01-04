// Initial wiring: [5, 19, 10, 11, 7, 16, 15, 3, 4, 0, 14, 18, 6, 1, 2, 12, 8, 9, 17, 13]
// Resulting wiring: [5, 19, 10, 11, 7, 16, 15, 3, 4, 0, 14, 18, 6, 1, 2, 12, 8, 9, 17, 13]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[5], q[4];
cx q[8], q[7];
cx q[10], q[9];
cx q[11], q[8];
cx q[12], q[7];
cx q[16], q[13];
cx q[17], q[16];
cx q[17], q[11];
cx q[12], q[18];
cx q[12], q[13];
cx q[7], q[12];
cx q[6], q[13];
cx q[13], q[14];
cx q[5], q[6];
cx q[3], q[4];
cx q[2], q[7];
cx q[1], q[7];
cx q[7], q[12];
cx q[12], q[18];
cx q[12], q[7];
