// Initial wiring: [7, 1, 2, 6, 8, 4, 0, 3, 5]
// Resulting wiring: [7, 1, 2, 6, 8, 4, 0, 3, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[7];
cx q[3], q[4];
cx q[4], q[7];
cx q[7], q[4];
cx q[6], q[7];
