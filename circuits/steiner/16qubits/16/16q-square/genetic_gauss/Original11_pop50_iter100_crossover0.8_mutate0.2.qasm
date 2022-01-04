// Initial wiring: [4, 5, 13, 6, 3, 11, 14, 12, 0, 10, 7, 1, 15, 8, 2, 9]
// Resulting wiring: [4, 5, 13, 6, 3, 11, 14, 12, 0, 10, 7, 1, 15, 8, 2, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[8], q[1];
cx q[9], q[2];
cx q[7], q[3];
cx q[8], q[4];
cx q[11], q[5];
cx q[15], q[14];
cx q[15], q[12];
cx q[12], q[0];
cx q[13], q[14];
cx q[8], q[11];
cx q[10], q[14];
cx q[4], q[13];
cx q[1], q[4];
cx q[0], q[1];
cx q[2], q[15];
