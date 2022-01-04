// EXPECTED_REWIRING [1 2 7 4 3 5 0 6 8 9 10 11 12 14 13 15]
// CURRENT_REWIRING [6 2 0 4 3 10 8 1 7 9 5 13 11 15 12 14]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[9];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-1.7843290499389812*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.077989633526896*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.8103884456122044*pi) q[10];
rz(0.59368010174542*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.958108965734335*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.4741817804854853*pi) q[1];
rz(-0.2135327231440851*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.077989633526896*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.8103884456122047*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.0537901828308989*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3844841619731472*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.27624762609369*pi) q[9];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[6];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(0.6463506146747173*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.495242038915076*pi) q[6];
cz q[6], q[5];
rz(-1.1645820567151615*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687766*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[14];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.9086943076984564*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.962833433305045*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.4016943021295236*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.3925074308386587*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.590663881884813*pi) q[12];
cz q[12], q[11];
rz(1.053987406345676*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
cz q[13], q[14];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(0.5936801017454187*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6015542728903499*pi) q[0];
rz(1.3572636036508106*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.0636030200628974*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.7604078811826913*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.1301913360488327*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.3294912647772694*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(1.7900205129423346*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.545514996826329*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.4578230667656809*pi) q[12];
rz(-1.784329049938982*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.077989633526895*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(1.333940690126882*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-2.381184772407101*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(1.5707963267948966*pi) q[14];
rz(-1.3294912647772712*pi) q[14];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.298994492700352*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268955*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-2.3811847724071007*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-1.1645820567151592*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.16538560610687794*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.730367851897572*pi) q[13];
cz q[14], q[13];
rz(1.0420735922095057*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.6134404140950419*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.873743426786342*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.623166354010596*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.0082828700362283*pi) q[8];
cz q[8], q[7];
rz(1.3419351522344822*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
cz q[3], q[4];
rz(1.4564375502462912*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4269954866939931*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[14];
rz(1.2183200733198107*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.21533785059204*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[8], q[15];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6463506146747164*pi) q[0];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.6463506146747164*pi) q[1];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[4];
rz(3.141592653589793*pi) q[5];
rz(0.2427132517316307*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.261599837637768*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.4681196075215537*pi) q[6];
rz(1.2740939557432278*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4895703198992696*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.7082561075355964*pi) q[7];
rz(0.7599720945337807*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(0.08197635545524928*pi) q[9];
rz(1.4564375502462912*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.426995486693993*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-3.059616298134544*pi) q[10];
rz(-0.9007671867299889*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.2063062546736758*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.2536859736107111*pi) q[11];
rz(-0.6542456812873576*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.9242262418970197*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.6463506146747164*pi) q[12];
rz(-0.1034406410691524*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(3.0114013175409617*pi) q[15];
rx(3.141592653589793*pi) q[15];
