// Initial wiring: [6 2 1 3 4 5 0 7 8]
// Resulting wiring: [6 2 1 3 4 5 0 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[4];
cx q[3], q[2];
cx q[1], q[0];
cx q[3], q[8];
cx q[6], q[7];
