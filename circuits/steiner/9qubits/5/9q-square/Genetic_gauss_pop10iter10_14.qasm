// Initial wiring: [0 4 1 3 5 2 6 7 8]
// Resulting wiring: [0 4 1 3 5 2 6 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[5];
cx q[5], q[0];
cx q[0], q[1];
cx q[3], q[4];
cx q[6], q[7];
