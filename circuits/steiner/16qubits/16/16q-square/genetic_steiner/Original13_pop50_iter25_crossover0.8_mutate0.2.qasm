// Initial wiring: [13, 0, 10, 12, 14, 9, 3, 5, 6, 4, 11, 2, 1, 7, 15, 8]
// Resulting wiring: [13, 0, 10, 12, 14, 9, 3, 5, 6, 4, 11, 2, 1, 7, 15, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[6], q[5];
cx q[8], q[7];
cx q[9], q[6];
cx q[12], q[11];
cx q[11], q[10];
cx q[12], q[11];
cx q[14], q[15];
cx q[9], q[14];
cx q[14], q[13];
cx q[6], q[9];
cx q[9], q[14];
cx q[14], q[15];
cx q[9], q[6];
cx q[4], q[5];
cx q[5], q[6];
cx q[6], q[9];
cx q[9], q[14];
cx q[3], q[4];
cx q[1], q[2];
