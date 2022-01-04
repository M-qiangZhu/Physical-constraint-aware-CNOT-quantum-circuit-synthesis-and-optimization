// Initial wiring: [7, 9, 8, 13, 6, 1, 17, 3, 2, 0, 11, 15, 19, 14, 12, 4, 18, 10, 5, 16]
// Resulting wiring: [7, 9, 8, 13, 6, 1, 17, 3, 2, 0, 11, 15, 19, 14, 12, 4, 18, 10, 5, 16]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[2], q[1];
cx q[10], q[9];
cx q[11], q[10];
cx q[12], q[11];
cx q[12], q[6];
cx q[14], q[5];
cx q[15], q[13];
cx q[17], q[12];
cx q[17], q[11];
cx q[14], q[15];
cx q[13], q[16];
cx q[16], q[17];
cx q[12], q[18];
cx q[11], q[18];
cx q[6], q[12];
cx q[5], q[6];
cx q[6], q[12];
cx q[12], q[11];
cx q[11], q[10];
cx q[12], q[6];
cx q[3], q[4];
cx q[2], q[8];
cx q[0], q[9];
