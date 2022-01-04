// EXPECTED_REWIRING [0 5 2 3 7 4 6 1 8]
// CURRENT_REWIRING [1 5 0 7 4 3 6 2 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(2.352814310306876*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.2197370482377071*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.13429310326690086*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.183321213886989*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.7966859589555537*pi) q[3];
cz q[3], q[2];
rz(2.1878261258005613*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.6993139092742953*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.7495252852366123*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.6285611251287253*pi) q[3];
cz q[3], q[4];
cz q[0], q[5];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.16576766742825289*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.7260112626204837*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[5];
rz(-2.1644764285403166*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.1834836878554581*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.667410873104308*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[4], q[1];
rz(0.8055270383770203*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.57770835091837*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.4835187068478887*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.23535866452886833*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.3024104870043598*pi) q[5];
cz q[5], q[0];
rz(2.934537351742044*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
cz q[3], q[8];
rz(1.674236967864048*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.9799474569139472*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.743864335899125*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.4396353927456323*pi) q[5];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.2313766178847327*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(1.0859990061624507*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.958108965734335*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.6015542728903503*pi) q[2];
rz(0.5752456201388326*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526896*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.3811847724071016*pi) q[3];
cz q[3], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(3.141592653589793*pi) q[8];
rx(-1.5707963267948966*pi) q[3];
rz(-1.4414518560841771*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.3844841619731474*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.4361413542909993*pi) q[4];
cz q[4], q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(2.761369489712264*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.9641888827222767*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.9438241621069082*pi) q[5];
rz(-0.9771162250494777*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.958108965734335*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.4741817804854858*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[5], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.1645820567151592*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687794*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.730367851897572*pi) q[5];
cz q[5], q[6];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.495242038915076*pi) q[1];
rz(1.7958397135536857*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5012906947992428*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.7927243225333314*pi) q[0];
cz q[5], q[0];
rz(-1.1645820567151568*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.1653856061068771*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[7];
cz q[1], q[0];
rz(1.0537901828308989*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731476*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.27624762609369*pi) q[7];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.2989944927003583*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.0779896335268955*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-2.381184772407101*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.9744254291867687*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.916578713362112*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.2753044905898345*pi) q[2];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[8];
rx(-1.5707963267948966*pi) q[4];
cz q[7], q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(-1.6336712147423624*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.5453597136422716*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.6508215738116782*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.135848094014972*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.06868118715190588*pi) q[2];
cz q[2], q[1];
rz(-0.9724236271544395*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(2.217146941469614*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.467355685725745*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-3.075736053375836*pi) q[3];
rz(0.33214998029491727*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.5823840341180073*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[4], q[1];
rz(0.4815132762225467*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(2.217146941469614*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.467355685725745*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-0.21353272314408464*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.077989633526896*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.3811847724071016*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(2.9280599304457087*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.0779896335268964*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.8103884456122046*pi) q[5];
rz(1.977010596874636*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.9762070474829154*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.4564375502462916*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.426995486693993*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(0.4112248016922231*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.4189783790674753*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.641679694489658*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0128514707776533*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.16554473130611172*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.7404910762766441*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.3654860930580351*pi) q[7];
cz q[7], q[4];
rz(-2.185532331713728*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(0.8953204217360743*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.1215451146451456*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.0427241473112119*pi) q[2];
rz(-2.960150229776044*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.2108011576186357*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[1], q[4];
rz(-0.5423120639166203*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5998329073868873*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.416846500246271*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.6314674336958177*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[0], q[5];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[5];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.495242038915076*pi) q[3];
rz(-2.9586092544615905*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.028484262878276*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[2];
rz(1.5424911671965855*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
rz(2.724765849357039*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[4];
rz(-3.102864715132582*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.3572636036508123*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.0636030200628979*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.7604078811826915*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(0.24271325173163064*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.261599837637768*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.244269372863136*pi) q[0];
cz q[5], q[0];
rz(-1.987623131027652*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[0];
rz(-1.1645820567151595*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.1653856061068779*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.5146654427613733*pi) q[1];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.028305159598310474*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rz(0.13691536220372194*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.8775449904435977*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.18790336066012708*pi) q[7];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
