// Initial wiring: [11, 7, 12, 5, 9, 8, 6, 10, 13, 14, 2, 15, 4, 1, 3, 0]
// Resulting wiring: [11, 7, 12, 5, 9, 8, 6, 10, 13, 14, 2, 15, 4, 1, 3, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[3], q[0];
cx q[4], q[2];
cx q[4], q[1];
cx q[5], q[1];
cx q[5], q[0];
cx q[6], q[3];
cx q[6], q[1];
cx q[7], q[6];
cx q[8], q[6];
cx q[8], q[4];
cx q[8], q[1];
cx q[9], q[7];
cx q[9], q[4];
cx q[9], q[1];
cx q[10], q[7];
cx q[10], q[4];
cx q[10], q[3];
cx q[10], q[1];
cx q[11], q[10];
cx q[11], q[7];
cx q[11], q[4];
cx q[8], q[0];
cx q[8], q[2];
cx q[7], q[5];
cx q[12], q[7];
cx q[12], q[5];
cx q[13], q[12];
cx q[13], q[9];
cx q[13], q[7];
cx q[13], q[5];
cx q[14], q[9];
cx q[14], q[7];
cx q[14], q[5];
cx q[15], q[9];
cx q[15], q[7];
cx q[9], q[0];
cx q[7], q[1];
cx q[5], q[2];
cx q[7], q[3];
cx q[14], q[6];
cx q[15], q[10];
cx q[13], q[11];
cx q[14], q[15];
cx q[12], q[14];
cx q[12], q[13];
cx q[13], q[12];
cx q[11], q[15];
cx q[11], q[14];
cx q[14], q[11];
cx q[10], q[14];
cx q[10], q[12];
cx q[9], q[14];
cx q[9], q[13];
cx q[9], q[12];
cx q[9], q[11];
cx q[8], q[15];
cx q[8], q[13];
cx q[8], q[10];
cx q[7], q[14];
cx q[7], q[13];
cx q[7], q[8];
cx q[6], q[11];
cx q[6], q[10];
cx q[6], q[8];
cx q[6], q[7];
cx q[5], q[9];
cx q[5], q[8];
cx q[5], q[7];
cx q[4], q[11];
cx q[4], q[8];
cx q[3], q[10];
cx q[3], q[7];
cx q[3], q[6];
cx q[3], q[4];
cx q[4], q[3];
cx q[2], q[13];
cx q[2], q[10];
cx q[2], q[9];
cx q[1], q[11];
cx q[1], q[7];
cx q[1], q[4];
cx q[1], q[2];
cx q[0], q[11];
cx q[0], q[9];
cx q[0], q[7];
cx q[0], q[3];
cx q[3], q[0];
cx q[6], q[15];
cx q[11], q[14];
cx q[2], q[12];
