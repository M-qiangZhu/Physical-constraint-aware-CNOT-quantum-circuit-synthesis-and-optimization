// Initial wiring: [0, 3, 7, 4, 8, 6, 5, 1, 2]
// Resulting wiring: [0, 3, 7, 4, 8, 6, 5, 1, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[8];
cx q[4], q[1];
cx q[7], q[4];
cx q[4], q[7];
