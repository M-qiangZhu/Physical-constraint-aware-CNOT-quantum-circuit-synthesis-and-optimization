// Initial wiring: [1, 6, 2, 7, 3, 4, 5, 0, 8]
// Resulting wiring: [1, 6, 2, 7, 3, 4, 5, 0, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[1], q[4];
cx q[6], q[7];
cx q[5], q[6];
cx q[4], q[7];
cx q[6], q[7];
cx q[3], q[8];
cx q[8], q[3];
cx q[3], q[2];
cx q[8], q[3];
cx q[4], q[1];
cx q[5], q[4];
