// Initial wiring: [0 1 3 2 7 5 4 6 8]
// Resulting wiring: [0 1 3 2 7 5 4 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[1];
cx q[7], q[8];
cx q[6], q[5];
cx q[2], q[3];
cx q[7], q[4];
