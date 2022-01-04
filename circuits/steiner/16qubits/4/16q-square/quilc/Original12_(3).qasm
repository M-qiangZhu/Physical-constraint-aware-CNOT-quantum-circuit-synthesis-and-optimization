// EXPECTED_REWIRING [0 6 2 3 14 4 1 7 8 9 11 5 12 13 10 15]
// CURRENT_REWIRING [0 6 2 3 14 4 1 7 8 9 11 5 12 13 10 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
cz q[5], q[10];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[15];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rz(3.141592653589793*pi) q[10];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
