// Initial wiring: [2 4 1 3 0 5 6 7 8]
// Resulting wiring: [2 4 1 3 0 5 6 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[5];
cx q[0], q[5];
cx q[7], q[6];
cx q[1], q[0];
cx q[7], q[8];
