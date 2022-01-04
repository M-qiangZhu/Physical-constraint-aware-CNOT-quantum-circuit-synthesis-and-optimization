// Initial wiring: [7, 8, 2, 5, 0, 1, 6, 4, 3]
// Resulting wiring: [7, 8, 2, 5, 0, 1, 6, 4, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[5], q[6];
cx q[4], q[5];
cx q[3], q[4];
cx q[2], q[3];
cx q[1], q[4];
cx q[3], q[4];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[7];
cx q[6], q[5];
cx q[7], q[6];
cx q[8], q[7];
cx q[6], q[7];
cx q[5], q[4];
cx q[8], q[3];
cx q[7], q[8];
