// Initial wiring: [8, 5, 4, 6, 3, 1, 0, 2, 7]
// Resulting wiring: [8, 5, 4, 6, 3, 1, 0, 2, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[7];
cx q[4], q[3];
cx q[3], q[2];
cx q[8], q[3];
cx q[3], q[8];
