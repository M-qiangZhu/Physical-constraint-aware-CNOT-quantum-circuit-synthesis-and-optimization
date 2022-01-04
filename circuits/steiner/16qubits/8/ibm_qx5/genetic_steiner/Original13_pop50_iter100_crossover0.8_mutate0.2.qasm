// Initial wiring: [7, 6, 13, 9, 0, 1, 2, 14, 10, 3, 5, 12, 4, 11, 8, 15]
// Resulting wiring: [7, 6, 13, 9, 0, 1, 2, 14, 10, 3, 5, 12, 4, 11, 8, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[4], q[3];
cx q[12], q[11];
cx q[14], q[1];
cx q[13], q[14];
cx q[3], q[12];
cx q[12], q[11];
cx q[2], q[3];
cx q[2], q[13];
cx q[3], q[12];
cx q[1], q[2];
cx q[2], q[13];
cx q[2], q[3];
