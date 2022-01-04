// EXPECTED_REWIRING [5 1 2 3 4 0 7 8 6]
// CURRENT_REWIRING [5 0 2 3 4 1 7 8 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[2];
rz(0.5936801017454187*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6015542728903499*pi) q[0];
rz(-0.21353272314408345*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.077989633526895*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.381184772407102*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.1645820567151592*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.16538560610687794*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.730367851897572*pi) q[1];
cz q[1], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
cz q[6], q[7];
rz(0.2427132517316307*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.261599837637768*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4681196075215537*pi) q[0];
rz(1.467355685725745*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rz(3.141592653589793*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(3.141592653589793*pi) q[7];
