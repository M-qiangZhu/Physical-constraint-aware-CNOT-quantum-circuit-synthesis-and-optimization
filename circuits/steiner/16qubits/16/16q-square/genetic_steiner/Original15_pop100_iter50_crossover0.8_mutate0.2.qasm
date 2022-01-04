// Initial wiring: [3, 11, 8, 7, 4, 1, 5, 6, 15, 14, 2, 12, 13, 10, 9, 0]
// Resulting wiring: [3, 11, 8, 7, 4, 1, 5, 6, 15, 14, 2, 12, 13, 10, 9, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[5];
cx q[8], q[7];
cx q[7], q[6];
cx q[8], q[7];
cx q[10], q[9];
cx q[11], q[4];
cx q[13], q[10];
cx q[12], q[13];
cx q[9], q[14];
cx q[7], q[8];
cx q[8], q[9];
cx q[8], q[7];
cx q[5], q[6];
cx q[5], q[10];
cx q[6], q[7];
cx q[2], q[3];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[3], q[2];
cx q[0], q[1];
