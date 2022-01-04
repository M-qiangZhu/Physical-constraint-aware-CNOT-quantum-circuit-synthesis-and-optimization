// Initial wiring: [4, 1, 6, 8, 5, 7, 12, 14, 9, 3, 2, 15, 11, 13, 0, 10]
// Resulting wiring: [4, 1, 6, 8, 5, 7, 12, 14, 9, 3, 2, 15, 11, 13, 0, 10]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[8], q[2];
cx q[9], q[6];
cx q[11], q[10];
cx q[11], q[8];
cx q[2], q[1];
cx q[7], q[3];
cx q[14], q[12];
cx q[12], q[13];
cx q[11], q[13];
cx q[6], q[13];
cx q[6], q[10];
cx q[6], q[15];
cx q[9], q[14];
cx q[8], q[12];
cx q[4], q[6];
cx q[2], q[4];
cx q[0], q[5];
cx q[0], q[4];
cx q[0], q[3];
cx q[0], q[2];
cx q[2], q[0];
cx q[1], q[13];
cx q[0], q[12];
cx q[1], q[10];
cx q[6], q[9];
cx q[3], q[8];
