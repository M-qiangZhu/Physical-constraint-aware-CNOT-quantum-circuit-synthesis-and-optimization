// Initial wiring: [8, 10, 5, 15, 12, 2, 14, 7, 3, 4, 9, 1, 0, 11, 13, 6]
// Resulting wiring: [8, 10, 5, 15, 12, 2, 14, 7, 3, 4, 9, 1, 0, 11, 13, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[3], q[2];
cx q[4], q[3];
cx q[10], q[5];
cx q[11], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[3], q[2];
cx q[4], q[3];
cx q[11], q[4];
cx q[12], q[11];
cx q[13], q[10];
cx q[10], q[13];
cx q[9], q[10];
cx q[10], q[13];
cx q[13], q[10];
cx q[8], q[15];
cx q[6], q[7];
cx q[3], q[4];
cx q[4], q[5];
