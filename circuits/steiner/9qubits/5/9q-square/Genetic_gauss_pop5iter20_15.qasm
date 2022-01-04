// Initial wiring: [0 1 3 2 7 5 6 8 4]
// Resulting wiring: [0 2 3 1 7 5 6 8 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[5], q[4];
cx q[7], q[6];
cx q[1], q[2];
cx q[1], q[2];
cx q[1], q[2];
cx q[1], q[0];
cx q[5], q[6];
