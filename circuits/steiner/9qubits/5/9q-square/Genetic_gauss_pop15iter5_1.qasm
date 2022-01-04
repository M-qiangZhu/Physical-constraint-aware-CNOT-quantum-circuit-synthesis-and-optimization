// Initial wiring: [5 2 1 3 4 6 0 7 8]
// Resulting wiring: [5 2 4 3 1 7 0 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[4], q[7];
cx q[2], q[1];
cx q[5], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[7], q[4];
