// Initial wiring: [1, 3, 4, 8, 2, 0, 6, 7, 5]
// Resulting wiring: [1, 3, 4, 8, 2, 0, 6, 7, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[2], q[3];
cx q[3], q[4];
cx q[4], q[5];
cx q[3], q[4];
cx q[2], q[3];
cx q[0], q[5];
cx q[4], q[5];
cx q[4], q[7];
cx q[3], q[4];
cx q[4], q[7];
cx q[7], q[8];
cx q[6], q[7];
cx q[3], q[8];
cx q[6], q[5];
cx q[7], q[4];
cx q[6], q[7];
cx q[2], q[1];
cx q[3], q[2];
