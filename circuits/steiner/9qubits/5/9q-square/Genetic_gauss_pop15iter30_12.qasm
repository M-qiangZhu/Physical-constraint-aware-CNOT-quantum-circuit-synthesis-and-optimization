// Initial wiring: [1 0 3 2 4 6 5 7 8]
// Resulting wiring: [1 0 3 2 4 6 5 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[1];
cx q[7], q[4];
cx q[7], q[6];
cx q[6], q[5];
cx q[3], q[2];
