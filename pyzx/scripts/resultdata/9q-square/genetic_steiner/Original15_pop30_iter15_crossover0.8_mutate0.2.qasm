// Initial wiring: [1, 3, 2, 0, 4, 6, 7, 5, 8]
// Resulting wiring: [1, 3, 2, 0, 4, 6, 7, 5, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[1], q[0];
cx q[8], q[7];
