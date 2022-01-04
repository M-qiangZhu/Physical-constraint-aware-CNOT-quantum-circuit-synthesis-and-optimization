// Initial wiring: [10, 5, 11, 6, 7, 9, 12, 13, 4, 1, 14, 2, 0, 3, 8, 15]
// Resulting wiring: [10, 5, 11, 6, 7, 9, 12, 13, 4, 1, 14, 2, 0, 3, 8, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[7], q[6];
cx q[6], q[5];
cx q[7], q[6];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[6];
cx q[11], q[4];
cx q[11], q[10];
cx q[12], q[11];
cx q[11], q[4];
cx q[12], q[11];
cx q[10], q[13];
cx q[2], q[3];
