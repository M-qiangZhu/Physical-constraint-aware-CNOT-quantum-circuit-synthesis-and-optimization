// Initial wiring: [0 4 2 3 5 1 6 7 8]
// Resulting wiring: [0 7 2 3 5 1 6 4 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[1];
cx q[2], q[1];
