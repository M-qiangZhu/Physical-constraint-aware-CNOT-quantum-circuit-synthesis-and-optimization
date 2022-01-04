// Initial wiring: [0, 13, 5, 3, 2, 14, 9, 4, 12, 11, 8, 7, 15, 6, 1, 10]
// Resulting wiring: [0, 13, 5, 3, 2, 14, 9, 4, 12, 11, 8, 7, 15, 6, 1, 10]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[3], q[2];
cx q[4], q[3];
cx q[11], q[4];
cx q[4], q[3];
cx q[11], q[4];
cx q[13], q[12];
cx q[13], q[2];
cx q[14], q[13];
cx q[15], q[14];
cx q[14], q[13];
cx q[13], q[12];
cx q[13], q[2];
cx q[14], q[1];
cx q[11], q[12];
cx q[10], q[11];
cx q[7], q[8];
cx q[6], q[9];
cx q[9], q[10];
cx q[2], q[13];
