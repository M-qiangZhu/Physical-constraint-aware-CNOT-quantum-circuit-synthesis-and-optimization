// Initial wiring: [0 1 2 3 7 4 6 5 8]
// Resulting wiring: [0 1 2 3 7 4 6 5 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[7];
cx q[8], q[7];
cx q[4], q[7];
