// Initial wiring: [5 4 2 8 1 0 7 6 3]
// Resulting wiring: [5 4 2 8 1 0 7 6 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[7], q[4];
cx q[8], q[3];
cx q[4], q[3];
cx q[5], q[4];
