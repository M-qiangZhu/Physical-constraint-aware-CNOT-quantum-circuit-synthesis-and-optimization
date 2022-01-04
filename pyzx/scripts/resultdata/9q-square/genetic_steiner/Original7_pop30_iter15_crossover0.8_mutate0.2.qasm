// Initial wiring: [1, 2, 4, 6, 0, 8, 7, 5, 3]
// Resulting wiring: [1, 2, 4, 6, 0, 8, 7, 5, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[3];
cx q[8], q[7];
cx q[7], q[6];
