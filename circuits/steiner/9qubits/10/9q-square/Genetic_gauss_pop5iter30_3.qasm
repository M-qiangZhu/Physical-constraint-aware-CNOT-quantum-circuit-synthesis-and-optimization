// Initial wiring: [0 1 2 3 4 6 7 5 8]
// Resulting wiring: [5 1 2 4 7 6 3 0 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[0];
cx q[5], q[4];
cx q[4], q[7];
cx q[3], q[8];
cx q[8], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[5], q[6];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[0], q[5];
cx q[0], q[5];
cx q[5], q[4];
cx q[1], q[4];
cx q[7], q[6];
