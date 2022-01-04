// Initial wiring: [11, 6, 15, 8, 7, 0, 13, 10, 5, 9, 1, 12, 3, 14, 2, 4]
// Resulting wiring: [11, 6, 15, 8, 7, 0, 13, 10, 5, 9, 1, 12, 3, 14, 2, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[5], q[4];
cx q[10], q[9];
cx q[11], q[10];
cx q[10], q[9];
cx q[11], q[10];
cx q[13], q[12];
cx q[14], q[13];
cx q[13], q[10];
cx q[14], q[13];
cx q[10], q[11];
cx q[6], q[9];
cx q[0], q[1];
