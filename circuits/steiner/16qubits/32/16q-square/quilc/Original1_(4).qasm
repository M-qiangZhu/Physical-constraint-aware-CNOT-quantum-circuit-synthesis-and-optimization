// EXPECTED_REWIRING [0 6 4 5 3 2 1 7 8 9 10 11 12 13 14 15]
// CURRENT_REWIRING [15 9 10 11 0 3 8 6 1 2 4 5 12 13 14 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(0.5936801017454187*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.958108965734335*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.6015542728903499*pi) q[3];
rz(1.3572636036508121*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.077989633526896*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-2.381184772407101*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
cz q[11], q[12];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.3480080966017534*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.2024946643615646*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rx(-1.5707963267948966*pi) q[5];
rz(0.6316983375666694*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-2.4322411164009146*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.1959887400036306*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.9156330120238814*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
rz(-2.928059930445705*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.077989633526896*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.3811847724071016*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
cz q[11], q[12];
rz(0.10344064106915161*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[3], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.495242038915076*pi) q[4];
rz(-1.1645820567151592*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.16538560610687794*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.730367851897572*pi) q[3];
cz q[4], q[3];
rz(-2.624586509625798*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.757108491616646*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.8653450274961032*pi) q[5];
rz(-2.364380883782938*pi) q[10];
rz(1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[13];
cz q[10], q[13];
rz(1.25382296258166*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.0779896335268964*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.8103884456122049*pi) q[11];
rz(0.10344064106915161*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.7843290499389812*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.0779896335268964*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.381184772407101*pi) q[14];
cz q[14], q[9];
rz(1.6366529270088535*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-2.087802470758894*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731474*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.2762476260936904*pi) q[7];
rz(-2.332931610504628*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.273057712884938*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.657730354091372*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(1.5707963267948966*pi) q[6];
rz(-1.4002406175966442*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(-1.1645820567151592*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687794*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.6269272108284194*pi) q[14];
rz(0.10344064106915161*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.19593203720896923*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.2519642934184128*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.8362546646201685*pi) q[6];
cz q[7], q[6];
rz(0.24685732763435286*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[9];
rz(-2.80031552953025*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.4266502937704122*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(0.10344064106915161*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[11], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(1.4564375502462896*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.426995486693991*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.487346972302432*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.2173664116927734*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.6097752594237011*pi) q[6];
cz q[5], q[6];
rz(-1.3108321476273328*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.1571396422585003*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.313895290890828*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-3.0818954748031064*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.9770105968746337*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.976207047482915*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-2.7303678518975714*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.7226142745223194*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.7843290499389812*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0779896335268964*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.381184772407101*pi) q[8];
cz q[8], q[7];
rz(-1.5049397265809397*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.0036142183255286*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268964*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.8103884456122046*pi) q[9];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(0.24271325173162997*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.2615998376377684*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.6734730460682392*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[14];
rz(-0.46738391902561993*pi) q[14];
rz(1.9770105968746339*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.976207047482916*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[14];
rz(1.366620450190422*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.8153458425107891*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.196439418140902*pi) q[5];
rz(-1.4189661656018442*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.2790553797743685*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rx(-1.5707963267948966*pi) q[5];
rz(-0.49114424200998497*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(-0.24491524571551282*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.130489234933764*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
cz q[14], q[15];
rz(1.2955005973476688*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.123273590281415*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.7322114912361384*pi) q[0];
rz(-0.18937090821039154*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.8771069984278869*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rx(-1.5707963267948966*pi) q[0];
rz(-1.6532625469239357*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[0], q[7];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(-2.2397648942608375*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.8438578786909945*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.15061011176837*pi) q[0];
rz(-0.8576340657541889*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.9179180121731327*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(-0.0010207176137582152*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-1.9175302048674592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.2845910486013002*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[8], q[7];
rz(0.9446340243167182*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(1.0537901828308989*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.3844841619731476*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.27624762609369*pi) q[15];
rz(-1.4527892815566474*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.5855007996414316*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.3236877800889277*pi) q[1];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.188626402044948*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.0801699305514028*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.7062524952048803*pi) q[5];
cz q[5], q[2];
rz(-0.9244457121201792*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(2.524872640235552*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.7571084916166462*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.8653450274961032*pi) q[5];
rz(3.141592653589793*pi) q[1];
rz(3.1050172983387676*pi) q[6];
rx(3.141592653589793*pi) q[6];
cz q[6], q[1];
rz(0.2102945010458361*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.892168000774516*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-3.0583465966373433*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.4986241205632305*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(3.089338179390804*pi) q[14];
cz q[14], q[9];
rz(1.3010927671321433*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-2.1801528629520837*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.6265833154835213*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4303780097612364*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.43070748769154554*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
cz q[15], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(1.674236967864048*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[5], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-0.40952570783419956*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7571084916166464*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.865345027496103*pi) q[10];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.495242038915076*pi) q[0];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
cz q[0], q[7];
rz(2.003614218325529*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0779896335268964*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.8103884456122046*pi) q[8];
rz(-1.4943768947841738*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4914298748732047*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(0.5936801017454187*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.958108965734335*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.6015542728903499*pi) q[13];
rz(0.056456052375446625*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.4506798445130826*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.6143237844423841*pi) q[14];
cz q[14], q[13];
rz(1.6366529270088535*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.1208713845982032*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.1645820567151592*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687794*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.730367851897572*pi) q[14];
rz(-1.1645820567151592*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.16538560610687794*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(2.730367851897572*pi) q[15];
cz q[14], q[15];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[5];
cz q[10], q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.343163130880458*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.0693316020761772*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[14];
cz q[14], q[9];
rz(-0.16372918275778484*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.9581089657343353*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(0.6015542728903515*pi) q[13];
rz(1.25382296258166*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.0779896335268964*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(1.6366529270088535*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-2.381184772407101*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.685155103343502*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7145971668958*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.341199133241645*pi) q[10];
rz(2.4873469723024355*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.2173664116927734*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(3.0381520125206407*pi) q[15];
rz(0.9244457121201792*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.1645820567151592*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687794*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.730367851897572*pi) q[14];
cz q[14], q[13];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
cz q[14], q[15];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[1];
rz(-0.6542456812873576*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9242262418970197*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.495242038915076*pi) q[2];
rz(3.0381520125206407*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.495242038915076*pi) q[4];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.495242038915076*pi) q[5];
rz(3.141592653589793*pi) q[6];
rz(2.217146941469614*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-1.1645820567151588*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687789*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.626927210828419*pi) q[8];
rz(1.5185132799312822*pi) q[9];
rz(-0.8823698758033984*pi) q[10];
rx(3.141592653589793*pi) q[10];
rz(1.4564375502462918*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.426995486693993*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-3.059616298134544*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(-1.5707963267948966*pi) q[13];
rz(1.467355685725745*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
