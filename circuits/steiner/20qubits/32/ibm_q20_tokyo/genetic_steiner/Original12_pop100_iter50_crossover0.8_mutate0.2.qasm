// Initial wiring: [12, 4, 3, 13, 14, 0, 8, 16, 5, 18, 15, 19, 17, 6, 1, 2, 10, 7, 11, 9]
// Resulting wiring: [12, 4, 3, 13, 14, 0, 8, 16, 5, 18, 15, 19, 17, 6, 1, 2, 10, 7, 11, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[3], q[2];
cx q[7], q[6];
cx q[8], q[2];
cx q[8], q[1];
cx q[9], q[8];
cx q[12], q[6];
cx q[6], q[5];
cx q[6], q[4];
cx q[6], q[3];
cx q[13], q[7];
cx q[7], q[1];
cx q[1], q[0];
cx q[13], q[7];
cx q[14], q[13];
cx q[13], q[7];
cx q[13], q[12];
cx q[7], q[1];
cx q[14], q[13];
cx q[15], q[13];
cx q[15], q[14];
cx q[13], q[7];
cx q[16], q[13];
cx q[13], q[7];
cx q[7], q[1];
cx q[1], q[0];
cx q[16], q[14];
cx q[7], q[1];
cx q[13], q[7];
cx q[17], q[12];
cx q[12], q[7];
cx q[18], q[12];
cx q[12], q[7];
cx q[19], q[18];
cx q[19], q[10];
cx q[18], q[12];
cx q[10], q[8];
cx q[17], q[18];
cx q[15], q[16];
cx q[14], q[15];
cx q[13], q[14];
cx q[12], q[18];
cx q[11], q[12];
cx q[10], q[11];
cx q[11], q[12];
cx q[9], q[11];
cx q[8], q[11];
cx q[11], q[18];
cx q[6], q[13];
cx q[13], q[14];
cx q[14], q[15];
cx q[14], q[13];
cx q[1], q[2];
cx q[0], q[9];
cx q[9], q[8];
