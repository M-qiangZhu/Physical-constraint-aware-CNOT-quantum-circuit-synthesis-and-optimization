// Initial wiring: [4, 10, 3, 1, 0, 6, 11, 5, 13, 2, 14, 9, 8, 12, 7, 15]
// Resulting wiring: [4, 10, 3, 1, 0, 6, 11, 5, 13, 2, 14, 9, 8, 12, 7, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[6], q[5];
cx q[15], q[14];
cx q[14], q[13];
cx q[13], q[12];
cx q[14], q[13];
cx q[11], q[12];
cx q[9], q[10];
cx q[8], q[9];
cx q[9], q[10];
cx q[9], q[8];
cx q[6], q[9];
cx q[6], q[7];
cx q[3], q[4];
cx q[4], q[3];
cx q[2], q[5];
