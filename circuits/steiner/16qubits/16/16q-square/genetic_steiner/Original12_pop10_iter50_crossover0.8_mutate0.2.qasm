// Initial wiring: [15, 3, 1, 0, 10, 8, 9, 6, 2, 14, 13, 12, 11, 7, 5, 4]
// Resulting wiring: [15, 3, 1, 0, 10, 8, 9, 6, 2, 14, 13, 12, 11, 7, 5, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[5];
cx q[9], q[8];
cx q[9], q[6];
cx q[10], q[9];
cx q[9], q[6];
cx q[10], q[9];
cx q[12], q[11];
cx q[11], q[10];
cx q[12], q[11];
cx q[15], q[14];
cx q[11], q[12];
cx q[12], q[13];
cx q[13], q[12];
cx q[10], q[13];
cx q[9], q[10];
cx q[10], q[11];
cx q[6], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[11], q[10];
cx q[5], q[10];
cx q[1], q[6];
cx q[6], q[9];
cx q[9], q[10];
cx q[10], q[13];
cx q[13], q[12];
cx q[9], q[8];
cx q[0], q[1];
cx q[1], q[6];
cx q[6], q[1];
