// Initial wiring: [9, 12, 3, 11, 10, 4, 13, 14, 0, 1, 6, 5, 7, 2, 15, 8]
// Resulting wiring: [9, 12, 3, 11, 10, 4, 13, 14, 0, 1, 6, 5, 7, 2, 15, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[5], q[2];
cx q[8], q[7];
cx q[10], q[5];
cx q[5], q[4];
cx q[4], q[3];
cx q[10], q[5];
cx q[11], q[4];
cx q[4], q[3];
cx q[13], q[12];
cx q[15], q[14];
cx q[14], q[9];
cx q[10], q[11];
cx q[6], q[9];
cx q[1], q[2];
cx q[2], q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[2], q[1];
