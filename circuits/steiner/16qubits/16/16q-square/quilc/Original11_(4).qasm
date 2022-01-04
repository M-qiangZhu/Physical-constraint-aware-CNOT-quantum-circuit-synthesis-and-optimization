// EXPECTED_REWIRING [6 2 7 4 14 9 11 1 0 8 5 3 12 10 13 15]
// CURRENT_REWIRING [5 2 0 6 14 8 11 1 7 9 4 3 12 10 13 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[10];
cz q[7], q[8];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
cz q[9], q[6];
rz(3.062148276737266*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.5395761164454771*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(0.5936801017454187*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6015542728903499*pi) q[4];
rz(1.3572636036508108*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.077989633526896*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(1.5707963267948966*pi) q[7];
rz(-1.4913519499423682*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(0.018303590245734396*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.6338472808168265*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5575761345092545*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.7237932273291712*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[6];
rz(1.6927167664697658*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[14], q[15];
rz(0.5936801017454182*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.958108965734335*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.474181780485487*pi) q[8];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(2.9280599304457073*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268964*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.3811847724071016*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[2], q[5];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[9], q[10];
rz(-1.031220210349417*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6463506146747164*pi) q[4];
rz(1.5840165190805373*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(0.2427132517316307*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.261599837637768*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.4681196075215537*pi) q[8];
rz(1.467355685725745*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(3.141592653589793*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(3.141592653589793*pi) q[12];
rz(1.5707963267948966*pi) q[14];
rx(3.141592653589793*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
