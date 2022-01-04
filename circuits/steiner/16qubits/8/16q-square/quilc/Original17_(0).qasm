// EXPECTED_REWIRING [7 6 2 3 5 4 1 0 8 9 11 10 13 12 14 15]
// CURRENT_REWIRING [8 6 2 3 5 4 1 0 10 7 11 9 13 12 14 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-3.075736053375836*pi) q[7];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(0.10344064106915161*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.087802470758894*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3844841619731474*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.4361413542909993*pi) q[9];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-2.087802470758894*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.3844841619731474*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.2762476260936904*pi) q[10];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[9];
cz q[10], q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.1645820567151588*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687789*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.626927210828419*pi) q[10];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.730367851897572*pi) q[8];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
cz q[13], q[10];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[6];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
cz q[5], q[10];
rz(2.217146941469614*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[14], q[9];
rz(3.141592653589793*pi) q[1];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rz(2.217146941469614*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-0.1034406410691524*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
