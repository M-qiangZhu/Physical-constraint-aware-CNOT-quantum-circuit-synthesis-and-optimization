// Initial wiring: [15, 6, 5, 10, 11, 12, 14, 7, 0, 2, 4, 9, 1, 13, 8, 3]
// Resulting wiring: [15, 6, 5, 10, 11, 12, 14, 7, 0, 2, 4, 9, 1, 13, 8, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[3], q[1];
cx q[3], q[0];
cx q[4], q[1];
cx q[4], q[0];
cx q[6], q[1];
cx q[8], q[4];
cx q[8], q[1];
cx q[10], q[9];
cx q[11], q[7];
cx q[11], q[4];
cx q[4], q[2];
cx q[11], q[3];
cx q[11], q[5];
cx q[13], q[12];
cx q[15], q[12];
cx q[13], q[1];
cx q[13], q[2];
cx q[15], q[4];
cx q[13], q[5];
cx q[13], q[8];
cx q[13], q[9];
cx q[9], q[10];
cx q[7], q[11];
cx q[6], q[7];
cx q[10], q[14];
cx q[7], q[12];
cx q[5], q[9];
cx q[4], q[9];
cx q[5], q[15];
cx q[9], q[12];
