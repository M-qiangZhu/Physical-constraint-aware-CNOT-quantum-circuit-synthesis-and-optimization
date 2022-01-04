// Initial wiring: [9, 12, 15, 6, 8, 7, 3, 14, 4, 10, 5, 2, 13, 0, 11, 1]
// Resulting wiring: [9, 12, 15, 6, 8, 7, 3, 14, 4, 10, 5, 2, 13, 0, 11, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[6], q[1];
cx q[7], q[6];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[6], q[1];
cx q[14], q[9];
cx q[14], q[15];
cx q[10], q[13];
cx q[9], q[14];
cx q[14], q[15];
cx q[15], q[14];
cx q[5], q[6];
cx q[4], q[5];
cx q[3], q[4];
cx q[2], q[3];
cx q[2], q[5];
cx q[3], q[4];
cx q[4], q[11];
cx q[5], q[6];
cx q[1], q[2];
cx q[0], q[7];
