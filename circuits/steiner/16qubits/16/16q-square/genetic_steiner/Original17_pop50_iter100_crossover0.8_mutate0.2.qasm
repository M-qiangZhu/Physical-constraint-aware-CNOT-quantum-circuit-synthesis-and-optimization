// Initial wiring: [6, 2, 7, 13, 9, 11, 10, 14, 3, 5, 0, 1, 4, 15, 12, 8]
// Resulting wiring: [6, 2, 7, 13, 9, 11, 10, 14, 3, 5, 0, 1, 4, 15, 12, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[5], q[2];
cx q[5], q[4];
cx q[2], q[1];
cx q[6], q[1];
cx q[10], q[9];
cx q[11], q[10];
cx q[11], q[4];
cx q[13], q[10];
cx q[14], q[13];
cx q[13], q[10];
cx q[15], q[14];
cx q[14], q[9];
cx q[15], q[14];
cx q[11], q[12];
cx q[10], q[11];
cx q[11], q[12];
cx q[7], q[8];
cx q[5], q[10];
cx q[10], q[11];
cx q[10], q[9];
cx q[3], q[4];
cx q[0], q[1];
