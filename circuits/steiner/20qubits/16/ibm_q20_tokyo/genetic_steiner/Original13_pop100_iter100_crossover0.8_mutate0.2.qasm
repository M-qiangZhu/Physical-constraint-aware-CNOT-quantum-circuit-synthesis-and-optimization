// Initial wiring: [3, 5, 19, 1, 7, 14, 0, 17, 16, 13, 6, 10, 2, 12, 4, 18, 15, 9, 11, 8]
// Resulting wiring: [3, 5, 19, 1, 7, 14, 0, 17, 16, 13, 6, 10, 2, 12, 4, 18, 15, 9, 11, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[6], q[4];
cx q[9], q[8];
cx q[10], q[8];
cx q[8], q[7];
cx q[8], q[2];
cx q[11], q[10];
cx q[12], q[7];
cx q[12], q[6];
cx q[13], q[7];
cx q[17], q[12];
cx q[12], q[6];
cx q[6], q[3];
cx q[18], q[19];
cx q[14], q[15];
cx q[13], q[15];
cx q[5], q[6];
cx q[3], q[6];
cx q[2], q[3];
cx q[3], q[6];
cx q[6], q[3];
