// Initial wiring: [5, 13, 10, 12, 14, 11, 0, 3, 15, 4, 1, 9, 6, 2, 7, 8]
// Resulting wiring: [5, 13, 10, 12, 14, 11, 0, 3, 15, 4, 1, 9, 6, 2, 7, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[3], q[2];
cx q[4], q[3];
cx q[3], q[2];
cx q[5], q[4];
cx q[5], q[2];
cx q[4], q[3];
cx q[2], q[1];
cx q[5], q[2];
cx q[5], q[4];
cx q[7], q[6];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[1];
cx q[8], q[7];
cx q[9], q[8];
cx q[9], q[6];
cx q[10], q[9];
cx q[9], q[6];
cx q[6], q[1];
cx q[10], q[5];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[8];
cx q[8], q[7];
cx q[10], q[9];
cx q[12], q[11];
cx q[11], q[10];
cx q[10], q[9];
cx q[14], q[9];
cx q[6], q[9];
cx q[1], q[6];
cx q[6], q[9];
cx q[9], q[6];
