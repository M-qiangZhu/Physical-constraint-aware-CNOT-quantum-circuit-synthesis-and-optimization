// Initial wiring: [2, 13, 11, 1, 10, 15, 7, 5, 0, 4, 12, 3, 9, 8, 14, 6]
// Resulting wiring: [2, 13, 11, 1, 10, 15, 7, 5, 0, 4, 12, 3, 9, 8, 14, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[5];
cx q[8], q[7];
cx q[11], q[10];
cx q[13], q[10];
cx q[14], q[15];
cx q[15], q[14];
cx q[10], q[13];
cx q[13], q[14];
cx q[7], q[8];
cx q[8], q[15];
cx q[15], q[14];
cx q[8], q[7];
cx q[5], q[10];
cx q[10], q[13];
cx q[13], q[10];
cx q[0], q[7];
cx q[7], q[8];
cx q[8], q[7];
