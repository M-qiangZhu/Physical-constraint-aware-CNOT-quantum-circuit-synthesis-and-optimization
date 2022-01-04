// Initial wiring: [0 1 3 2 4 6 5 7 8]
// Resulting wiring: [0 1 4 2 3 6 5 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[4], q[7];
cx q[0], q[5];
cx q[3], q[2];
cx q[7], q[8];
