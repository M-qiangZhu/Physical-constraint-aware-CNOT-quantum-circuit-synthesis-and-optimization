// Initial wiring: [0 1 2 3 4 5 6 8 7]
// Resulting wiring: [0 1 7 2 3 5 6 8 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[5], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[5], q[6];
cx q[6], q[5];
cx q[3], q[2];
cx q[3], q[2];
cx q[3], q[2];
cx q[5], q[4];
cx q[3], q[8];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[1], q[4];
cx q[6], q[7];
