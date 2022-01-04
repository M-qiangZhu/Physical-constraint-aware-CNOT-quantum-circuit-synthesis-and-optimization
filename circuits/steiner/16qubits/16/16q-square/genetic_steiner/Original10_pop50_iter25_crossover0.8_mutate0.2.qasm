// Initial wiring: [14, 12, 15, 6, 0, 1, 4, 7, 3, 9, 2, 10, 11, 5, 13, 8]
// Resulting wiring: [14, 12, 15, 6, 0, 1, 4, 7, 3, 9, 2, 10, 11, 5, 13, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[9], q[6];
cx q[10], q[9];
cx q[9], q[6];
cx q[6], q[1];
cx q[10], q[9];
cx q[12], q[11];
cx q[11], q[10];
cx q[10], q[9];
cx q[14], q[13];
cx q[13], q[12];
cx q[14], q[9];
cx q[15], q[14];
cx q[7], q[8];
cx q[8], q[7];
cx q[5], q[10];
cx q[10], q[9];
cx q[1], q[2];
