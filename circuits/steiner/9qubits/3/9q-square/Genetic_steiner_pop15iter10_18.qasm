// Initial wiring: [7, 4, 5, 2, 0, 6, 3, 1, 8]
// Resulting wiring: [7, 4, 5, 2, 0, 6, 3, 1, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[3];
cx q[4], q[1];
cx q[1], q[0];
