// Initial wiring: [0 4 2 3 5 6 1 7 8]
// Resulting wiring: [0 4 2 3 5 7 1 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[5], q[0];
cx q[4], q[5];
cx q[7], q[8];
