// Initial wiring: [1, 8, 5, 6, 3, 4, 7, 0, 2]
// Resulting wiring: [1, 8, 5, 6, 3, 4, 7, 0, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[7], q[4];
cx q[3], q[4];
