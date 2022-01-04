// Initial wiring: [14, 9, 3, 7, 6, 12, 15, 4, 13, 2, 8, 11, 10, 1, 5, 0]
// Resulting wiring: [14, 9, 3, 7, 6, 12, 15, 4, 13, 2, 8, 11, 10, 1, 5, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[5], q[2];
cx q[7], q[6];
cx q[6], q[5];
cx q[7], q[0];
cx q[7], q[6];
cx q[10], q[5];
cx q[5], q[2];
cx q[10], q[9];
cx q[10], q[5];
cx q[12], q[11];
cx q[13], q[12];
cx q[12], q[11];
cx q[14], q[15];
cx q[11], q[12];
cx q[12], q[13];
cx q[9], q[10];
cx q[6], q[9];
cx q[9], q[10];
cx q[5], q[6];
cx q[2], q[3];
