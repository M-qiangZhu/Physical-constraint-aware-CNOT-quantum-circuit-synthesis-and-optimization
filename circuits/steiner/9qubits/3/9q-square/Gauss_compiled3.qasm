// Initial wiring: [0 1 2 3 7 5 6 4 8]
// Resulting wiring: [0 1 2 3 7 5 6 4 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[5];
cx q[4], q[5];
cx q[3], q[8];
