// Initial wiring: [0 1 2 3 4 5 6 8 7]
// Resulting wiring: [0 1 2 3 4 5 6 8 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[8], q[3];
cx q[7], q[6];
