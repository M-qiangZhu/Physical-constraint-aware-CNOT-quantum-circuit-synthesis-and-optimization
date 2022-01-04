// Initial wiring: [6 1 2 3 4 0 5 7 8]
// Resulting wiring: [7 1 2 3 4 0 5 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[7];
cx q[5], q[4];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[0], q[1];
cx q[5], q[6];
cx q[3], q[8];
