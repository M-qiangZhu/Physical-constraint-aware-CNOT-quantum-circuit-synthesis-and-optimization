// EXPECTED_REWIRING [7 1 2 3 4 5 6 0 9 13 11 8 12 14 10 15]
// CURRENT_REWIRING [8 7 1 2 0 10 5 11 6 15 4 9 12 14 3 13]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-2.087802470758894*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.3844841619731474*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.2762476260936904*pi) q[14];
rz(0.59368010174542*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4741817804854853*pi) q[4];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[5];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(1.674236967864048*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.4189783790674746*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[14], q[13];
rz(1.6366529270088535*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.077989633526896*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.8103884456122044*pi) q[15];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[6];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(-0.21353272314408464*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.077989633526896*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.3811847724071016*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-2.087802470758894*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.3844841619731474*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.2762476260936904*pi) q[10];
rz(-2.6625757902999436*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.8385954038498077*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.387104966695441*pi) q[4];
rz(-2.087802470758894*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.3844841619731474*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.2762476260936904*pi) q[11];
rz(1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[14];
cz q[15], q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[10], q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(0.59368010174542*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.958108965734335*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.4741817804854853*pi) q[9];
rz(2.761369489712264*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.9641888827222767*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.9438241621069082*pi) q[10];
rz(1.4564375502462912*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.426995486693993*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-3.059616298134544*pi) q[11];
rz(1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
cz q[11], q[12];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.495242038915076*pi) q[9];
rz(-0.6542456812873576*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.9242262418970197*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.495242038915076*pi) q[14];
rz(2.217146941469614*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[6], q[5];
cz q[9], q[14];
rz(-0.6542456812873574*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970194*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(2.438433134961103*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.79815778156478*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.740466368626876*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.0719808591146385*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.4283283534793547*pi) q[11];
cz q[11], q[10];
rz(-0.8565730096786428*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(3.141592653589793*pi) q[3];
rz(0.0526705129292977*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.18348368785545796*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.6832977533602598*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.2397778643584514*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.8070403445577465*pi) q[11];
cz q[11], q[4];
rz(-0.9033854536905892*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.638842731429806*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.1645820567151595*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.1653856061068779*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.5146654427613733*pi) q[11];
rz(-3.0381520125206416*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.075736053375836*pi) q[2];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(2.478303775947115*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0012084816670503*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.3103278487967365*pi) q[10];
cz q[11], q[12];
rz(2.217146941469614*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-1.784329049938982*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.077989633526895*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-2.381184772407101*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[5];
rz(0.16485871111411673*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8907780568673315*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.5470206892193081*pi) q[2];
rz(3.1163997734097735*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.5545739736862436*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(3.075017285545588*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(-1.7157533645742222*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4530033030988418*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.2486041751633508*pi) q[2];
rz(1.3572636036508126*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.0636030200628968*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-0.7411594229350493*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.760407881182692*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.784329049938982*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268955*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-2.381184772407101*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-2.6625757902999436*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8385954038498077*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.387104966695441*pi) q[2];
rz(-2.725675911902684*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.7100898446287491*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.4480481373444407*pi) q[3];
rz(-1.1645820567151592*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687794*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.730367851897572*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(0.5936801017454187*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6015542728903499*pi) q[0];
rz(-0.21353272314408533*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.077989633526896*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.381184772407101*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.5707963267948966*pi) q[4];
rx(3.141592653589793*pi) q[4];
rz(-0.1034406410691524*pi) q[5];
rz(-1.1645820567151595*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.1653856061068779*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.5146654427613733*pi) q[1];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(-2.4058433798480308*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.3902282350570143*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.5213949060824536*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.7916044455469486*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.3728412324330104*pi) q[7];
cz q[7], q[6];
rz(1.416941371584262*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.6424590167724*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.22792738644686*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[9], q[6];
rz(1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.674236967864048*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[14], q[9];
rz(-2.3918013978459243*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.7160297807521157*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.1576744933903957*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.8167159576682943*pi) q[7];
cz q[7], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.8257604955356967*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.4564375502462912*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.4269954866939931*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-2.7522253481514634*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(3.0245775394268657*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.3590102639103083*pi) q[13];
rz(-1.6771388893750367*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.7670396961732827*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(-1.5707963267948966*pi) q[13];
rz(0.09038517843615423*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[13], q[14];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(0.08197635545524928*pi) q[15];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
cz q[5], q[4];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(1.467355685725745*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-0.38332567306041504*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.6417381409901057*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.8601062257635363*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5413409713512076*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.07755542150960397*pi) q[9];
cz q[9], q[6];
rz(-1.0014811236146413*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.8050904693551664*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.28948914855079216*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.8455327258820278*pi) q[6];
rz(1.674236967864048*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.4404711728706716*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.0483401401115116*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.1720909361479912*pi) q[9];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-3.1327667083915216*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-0.6542456812873576*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970197*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.6463506146747173*pi) q[10];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[9], q[10];
rz(1.984361829689743*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.7571084916166466*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.8653450274961029*pi) q[11];
rz(-1.5999566948607118*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0563694397464831*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.3964654582368251*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5841489492511402*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.4690855254999886*pi) q[5];
cz q[5], q[2];
rz(1.7080016415399797*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(2.4873469723024364*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.217366411692775*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.280193199370182*pi) q[1];
cz q[6], q[1];
rz(-2.854130120527456*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.749953171304915*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.46798964887298844*pi) q[4];
rz(-0.5352605475089192*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.988862680510689*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(-1.5707963267948966*pi) q[4];
rz(-1.7503679501115084*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.2114646402951834*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7434970087063815*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.9875118385806452*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.3622195608442609*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.7720663079244476*pi) q[11];
cz q[11], q[10];
rz(-1.2332079259041375*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(3.141592653589793*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-0.2643800298106018*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.3630558716977355*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.7842263001285642*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.0792166735486424*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.5487744413575033*pi) q[11];
cz q[11], q[4];
rz(-1.226698776426834*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-3.077185319397108*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.8966633308048995*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.087802470758895*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.384484161973147*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(2.9655132055310647*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.4361413542909993*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(2.974736504409638*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.1245016771314396*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.9260058160900922*pi) q[4];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(1.2152976607513999*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.15038009454217663*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.3592180369336142*pi) q[11];
cz q[12], q[11];
cz q[4], q[11];
rz(-1.2431963851356644*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4737797884101296*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rz(-1.4701256302314052*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.435985115313914*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[15];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.8058002501290371*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.6334799369811541*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.015330085059770427*pi) q[13];
cz q[12], q[13];
rz(0.2427132517316307*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.261599837637768*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4681196075215537*pi) q[0];
rz(-2.9265438140448987*pi) q[1];
rx(3.141592653589793*pi) q[1];
rz(-3.0108283919360117*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(3.0390121272822213*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.0973463976374846*pi) q[2];
rz(-1.1645820567151592*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.16538560610687794*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.6269272108284194*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.6463506146747173*pi) q[5];
rz(-1.1645820567151595*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.1653856061068779*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.5146654427613733*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.6463506146747164*pi) q[8];
rz(1.467355685725745*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-2.1091536791461536*pi) q[10];
rz(3.136578481852432*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(3.141592653589793*pi) q[12];
rz(1.5142835552549165*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(2.4815249967789477*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
