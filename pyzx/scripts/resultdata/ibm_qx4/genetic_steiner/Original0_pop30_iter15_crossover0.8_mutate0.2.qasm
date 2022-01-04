// Initial wiring: [0, 4, 3, 1, 2]
// Resulting wiring: [0, 4, 3, 1, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[1], q[0];
cx q[4], q[3];
cx q[2], q[3];
