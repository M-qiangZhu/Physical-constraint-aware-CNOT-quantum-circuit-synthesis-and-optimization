// Initial wiring: [2, 13, 14, 5, 4, 1, 0, 9, 8, 7, 10, 11, 6, 3, 15, 12]
// Resulting wiring: [2, 13, 14, 5, 4, 1, 0, 9, 8, 7, 10, 11, 6, 3, 15, 12]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[3], q[2];
cx q[4], q[3];
cx q[11], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[11], q[4];
cx q[13], q[12];
cx q[13], q[10];
cx q[14], q[9];
cx q[9], q[8];
cx q[15], q[8];
cx q[11], q[12];
cx q[7], q[8];
cx q[5], q[6];
cx q[6], q[5];
cx q[1], q[2];
cx q[0], q[1];
