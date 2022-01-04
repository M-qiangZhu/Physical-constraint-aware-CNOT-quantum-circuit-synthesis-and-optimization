// Initial wiring: [12, 4, 13, 2, 15, 1, 10, 0, 14, 6, 5, 3, 7, 8, 11, 9]
// Resulting wiring: [12, 4, 13, 2, 15, 1, 10, 0, 14, 6, 5, 3, 7, 8, 11, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[10], q[5];
cx q[5], q[2];
cx q[10], q[5];
cx q[11], q[10];
cx q[11], q[4];
cx q[12], q[11];
cx q[11], q[4];
cx q[13], q[12];
cx q[15], q[14];
cx q[14], q[13];
cx q[9], q[10];
cx q[6], q[9];
cx q[9], q[10];
cx q[10], q[9];