// Initial wiring: [0 1 3 8 4 7 5 6 2]
// Resulting wiring: [0 2 3 8 4 7 5 6 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[0];
cx q[4], q[1];
cx q[1], q[2];
cx q[1], q[2];
cx q[1], q[2];
cx q[0], q[1];
cx q[5], q[6];
cx q[3], q[4];
