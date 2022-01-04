// Initial wiring: [3, 7, 8, 2, 12, 11, 0, 15, 9, 4, 13, 5, 10, 14, 1, 6]
// Resulting wiring: [3, 7, 8, 2, 12, 11, 0, 15, 9, 4, 13, 5, 10, 14, 1, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[5], q[4];
cx q[7], q[6];
cx q[6], q[5];
cx q[8], q[7];
cx q[7], q[0];
cx q[9], q[6];
cx q[13], q[12];
cx q[13], q[10];
cx q[14], q[9];
cx q[9], q[6];
cx q[6], q[5];
cx q[15], q[14];
cx q[14], q[9];
cx q[9], q[6];
cx q[15], q[8];
cx q[6], q[9];
cx q[5], q[10];
cx q[10], q[9];
cx q[1], q[6];
cx q[6], q[9];
