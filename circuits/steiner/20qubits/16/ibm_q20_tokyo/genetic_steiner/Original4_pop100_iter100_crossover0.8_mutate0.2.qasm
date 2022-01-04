// Initial wiring: [2, 13, 17, 8, 16, 3, 7, 9, 11, 1, 0, 12, 5, 10, 14, 19, 15, 4, 18, 6]
// Resulting wiring: [2, 13, 17, 8, 16, 3, 7, 9, 11, 1, 0, 12, 5, 10, 14, 19, 15, 4, 18, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[1], q[0];
cx q[4], q[3];
cx q[5], q[3];
cx q[12], q[11];
cx q[11], q[10];
cx q[11], q[9];
cx q[12], q[7];
cx q[15], q[14];
cx q[16], q[13];
cx q[16], q[15];
cx q[13], q[7];
cx q[18], q[12];
cx q[7], q[8];
cx q[2], q[8];
cx q[2], q[3];
cx q[1], q[7];
