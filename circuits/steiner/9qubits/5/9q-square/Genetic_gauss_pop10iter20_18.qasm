// Initial wiring: [4 1 2 3 6 0 5 7 8]
// Resulting wiring: [4 1 2 3 6 0 5 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[8], q[3];
cx q[3], q[4];
cx q[4], q[5];
cx q[4], q[3];
cx q[8], q[7];
