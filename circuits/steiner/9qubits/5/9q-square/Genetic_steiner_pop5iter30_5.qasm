// Initial wiring: [7, 8, 5, 4, 3, 1, 2, 0, 6]
// Resulting wiring: [7, 8, 5, 4, 3, 1, 2, 0, 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[8], q[3];
cx q[3], q[2];
cx q[8], q[3];
cx q[3], q[8];
cx q[4], q[1];
cx q[5], q[4];
cx q[5], q[0];
