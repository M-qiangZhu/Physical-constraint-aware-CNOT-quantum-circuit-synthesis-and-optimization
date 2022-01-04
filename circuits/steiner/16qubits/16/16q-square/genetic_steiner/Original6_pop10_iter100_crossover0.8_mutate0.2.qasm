// Initial wiring: [4, 10, 11, 2, 12, 5, 6, 0, 1, 3, 14, 13, 8, 15, 7, 9]
// Resulting wiring: [4, 10, 11, 2, 12, 5, 6, 0, 1, 3, 14, 13, 8, 15, 7, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[4], q[3];
cx q[6], q[1];
cx q[8], q[7];
cx q[9], q[6];
cx q[6], q[1];
cx q[11], q[4];
cx q[11], q[10];
cx q[4], q[3];
cx q[15], q[14];
cx q[14], q[13];
cx q[13], q[12];
cx q[14], q[13];
cx q[15], q[14];
cx q[13], q[14];
cx q[7], q[8];
cx q[5], q[6];
cx q[4], q[11];
cx q[11], q[12];
cx q[3], q[4];
cx q[4], q[11];
cx q[11], q[4];
cx q[2], q[5];
cx q[5], q[6];
cx q[2], q[3];
