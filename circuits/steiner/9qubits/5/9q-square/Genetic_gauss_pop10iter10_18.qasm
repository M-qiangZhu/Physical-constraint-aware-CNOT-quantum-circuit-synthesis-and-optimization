// Initial wiring: [5 4 0 3 2 1 6 7 8]
// Resulting wiring: [5 4 0 3 2 1 6 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[4], q[5];
cx q[5], q[0];
cx q[5], q[4];
cx q[7], q[8];
