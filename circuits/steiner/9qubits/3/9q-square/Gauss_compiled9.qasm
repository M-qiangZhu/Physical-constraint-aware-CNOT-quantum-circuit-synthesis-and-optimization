// Initial wiring: [0 1 3 2 4 5 6 7 8]
// Resulting wiring: [0 1 3 2 4 5 7 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[3];
cx q[7], q[6];
cx q[7], q[6];
cx q[7], q[6];
cx q[4], q[7];
