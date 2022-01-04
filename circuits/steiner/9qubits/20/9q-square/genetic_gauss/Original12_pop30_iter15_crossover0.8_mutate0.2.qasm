// Initial wiring: [7, 3, 6, 5, 4, 1, 2, 8, 0]
// Resulting wiring: [7, 3, 6, 5, 4, 1, 2, 8, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[2], q[1];
cx q[3], q[0];
cx q[5], q[4];
cx q[5], q[3];
cx q[5], q[1];
cx q[5], q[0];
cx q[6], q[3];
cx q[7], q[3];
cx q[8], q[6];
cx q[6], q[1];
cx q[6], q[5];
cx q[3], q[4];
cx q[2], q[7];
cx q[1], q[7];
cx q[1], q[4];
cx q[5], q[8];
