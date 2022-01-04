// Initial wiring: [3, 0, 2, 5, 8, 7, 4, 6, 1]
// Resulting wiring: [3, 0, 2, 5, 8, 7, 4, 6, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[7];
cx q[7], q[8];
cx q[6], q[5];
cx q[4], q[1];
cx q[2], q[1];
