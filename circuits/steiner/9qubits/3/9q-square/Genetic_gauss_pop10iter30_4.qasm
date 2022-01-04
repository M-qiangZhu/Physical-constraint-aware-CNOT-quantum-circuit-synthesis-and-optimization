// Initial wiring: [0 1 2 3 5 4 6 7 8]
// Resulting wiring: [0 1 3 2 5 4 6 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[8], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[2], q[3];
cx q[3], q[8];
cx q[3], q[4];
