// EXPECTED_REWIRING [0 1 2 5 4 3 6 13 8 11 10 17 12 15 14 7 9 16 18 19]
// CURRENT_REWIRING [0 1 2 5 4 3 6 13 8 11 10 17 12 15 14 7 9 16 18 19]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[9];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[7];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rz(3.141592653589793*pi) q[7];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
