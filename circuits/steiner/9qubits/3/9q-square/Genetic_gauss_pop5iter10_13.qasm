// Initial wiring: [0 1 2 3 4 6 7 8 5]
// Resulting wiring: [0 1 2 3 4 6 7 8 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[0];
cx q[5], q[6];
cx q[4], q[7];
