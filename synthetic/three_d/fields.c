static const struct {
    u64 threshold;
    double width, pdf;
} syn_ziggurat[256] = {
    {0xef33d8025ef6a, 0x1.f493b7815d979p-51, 0x1.0000000000000p+0},
    {0x0, 0x1.b8d0be3fdf6c6p-55, 0x1.f446ac979f087p-1},
    {0xc08be98fbc6a8, 0x1.250af3c2c5bb4p-54, 0x1.eb7545b6ca915p-1},
    {0xda354fabd8142, 0x1.57cb938443b61p-54, 0x1.e3f11e027f077p-1},
    {0xe51f67ec1eeea, 0x1.801fce82fa70cp-54, 0x1.dd36fa704de95p-1},
    {0xeb255e9d3f77e, 0x1.a230c2e4cd0bcp-54, 0x1.d70920657bcf2p-1},
    {0xeef4b817ecab9, 0x1.c004d2f3861f7p-54, 0x1.d144978a119dcp-1},
    {0xf19470afa44aa, 0x1.dac2f5a747274p-54, 0x1.cbd33a8a72debp-1},
    {0xf37ed61ffcb18, 0x1.f32482d4cd5c3p-54, 0x1.c6a5ecea9787fp-1},
    {0xf4f469561255c, 0x1.04d32278ebbadp-53, 0x1.c1b1cd9eebaeap-1},
    {0xf61a5e41ba396, 0x1.0f5053b025d43p-53, 0x1.bceeb4ee1dc82p-1},
    {0xf707a755396a4, 0x1.192a697413677p-53, 0x1.b85653a8ff552p-1},
    {0xf7cb2ec28449a, 0x1.227a28f7a1af5p-53, 0x1.b3e3a8234dd10p-1},
    {0xf86f10c6357d3, 0x1.2b52e3863d880p-53, 0x1.af92a3f6ce8a2p-1},
    {0xf8fa6578325de, 0x1.33c3fc05791f5p-53, 0x1.ab5fef17a2504p-1},
    {0xf9724c74dd0da, 0x1.3bd9ec1a2b12fp-53, 0x1.a748bd550c9e1p-1},
    {0xf9da907dbf509, 0x1.439ef8dff9b55p-53, 0x1.a34aafdf5af0fp-1},
    {0xfa360f581fa74, 0x1.4b1bb363dfea7p-53, 0x1.9f63bee651fd8p-1},
    {0xfa86fde5b4bf8, 0x1.52575621ad374p-53, 0x1.9b9228d240681p-1},
    {0xfacf160d354dc, 0x1.59580a707ce96p-53, 0x1.97d4657617ac1p-1},
    {0xfb0fb6718b90f, 0x1.60231cfd97eeap-53, 0x1.94291c21b7a47p-1},
    {0xfb49f8d5374c6, 0x1.66bd261a37c3dp-53, 0x1.908f1bd31714fp-1},
    {0xfb7ec2366fe77, 0x1.6d2a292000570p-53, 0x1.8d0554fe60aa8p-1},
    {0xfbaece9a1e50e, 0x1.736dad346f8a6p-53, 0x1.898ad48badf02p-1},
    {0xfbdab9d040bed, 0x1.798ad10b32a77p-53, 0x1.861ebfc37bcacp-1},
    {0xfc03060ff6c57, 0x1.7f845ad46f543p-53, 0x1.82c050f56cf6ep-1},
    {0xfc2821037a248, 0x1.855cc53430a77p-53, 0x1.7f6ed4b20e2cbp-1},
    {0xfc4a67ae25bd1, 0x1.8b1649e7b769ap-53, 0x1.7c29a779c6858p-1},
    {0xfc6a2977aee31, 0x1.90b2ea94ecf98p-53, 0x1.78f033ca0b0d5p-1},
    {0xfc87aa92896a4, 0x1.96347822c1eeap-53, 0x1.75c1f0770d856p-1},
    {0xfca325e4bde85, 0x1.9b9c98e38c546p-53, 0x1.729e5f43f6d12p-1},
    {0xfcbcce902231a, 0x1.a0eccdca4a72cp-53, 0x1.6f850baea7aeep-1},
    {0xfcd4d12f839c4, 0x1.a62676d77cd59p-53, 0x1.6c7589e635a89p-1},
    {0xfceb54d8fec99, 0x1.ab4ad6e101630p-53, 0x1.696f75e513b2ap-1},
    {0xfd007bf1dc930, 0x1.b05b16d136c9cp-53, 0x1.667272a92e323p-1},
    {0xfd1464dd6c4e6, 0x1.b558487427a29p-53, 0x1.637e298550c18p-1},
    {0xfd272a8e2f450, 0x1.ba4368e529f3ap-53, 0x1.6092498802665p-1},
    {0xfd38e4ff0c91e, 0x1.bf1d62abf8232p-53, 0x1.5dae86f4aff6ap-1},
    {0xfd49a9990b478, 0x1.c3e70f9594ef3p-53, 0x1.5ad29acc85c89p-1},
    {0xfd598b8920f53, 0x1.c8a13a5323b61p-53, 0x1.57fe4264c8d8fp-1},
    {0xfd689c08e99ec, 0x1.cd4c9fe72268bp-53, 0x1.55313f08d9e46p-1},
    {0xfd76ea9c8e832, 0x1.d1e9f0e80b748p-53, 0x1.526b55a656cd5p-1},
    {0xfd848547b08e8, 0x1.d679d29e41f10p-53, 0x1.4fac4e820b667p-1},
    {0xfd9178bad2c8c, 0x1.dafce0023b8c3p-53, 0x1.4cf3f4f494ec0p-1},
    {0xfd9dd07a7add2, 0x1.df73aa9f17653p-53, 0x1.4a42172dc5278p-1},
    {0xfda9970105e8c, 0x1.e3debb5d2edfep-53, 0x1.479685fdf5012p-1},
    {0xfdb4d5dc02e20, 0x1.e83e9337a6f00p-53, 0x1.44f114a493679p-1},
    {0xfdbf95c5bfcd0, 0x1.ec93abdf982cep-53, 0x1.425198a355fe3p-1},
    {0xfdc9debb99a7d, 0x1.f0de784f06226p-53, 0x1.3fb7e99585b82p-1},
    {0xfdd3b8118729d, 0x1.f51f654d8f688p-53, 0x1.3d23e10af31a3p-1},
    {0xfddd288342f90, 0x1.f956d9e87d7aep-53, 0x1.3a955a662cd0ep-1},
    {0xfde6364369f64, 0x1.fd8537dfa2eacp-53, 0x1.380c32bda00d5p-1},
    {0xfdeee708d514e, 0x1.00d56e04234ecp-52, 0x1.358848bf550e9p-1},
    {0xfdf7401a6b42e, 0x1.02e40f5398f9ap-52, 0x1.33097c9703a35p-1},
    {0xfdff46599ed40, 0x1.04eea9e16a5fcp-52, 0x1.308fafd6438efp-1},
    {0xfe06fe4bc24f2, 0x1.06f565b72a010p-52, 0x1.2e1ac55ea3beep-1},
    {0xfe0e6c225a258, 0x1.08f869071f40bp-52, 0x1.2baaa14d7954ap-1},
    {0xfe1593c28b84c, 0x1.0af7d84bc6113p-52, 0x1.293f28e93cd15p-1},
    {0xfe1c78cbc3f99, 0x1.0cf3d664bcc7fp-52, 0x1.26d84290504edp-1},
    {0xfe231e9db1caa, 0x1.0eec84b16086bp-52, 0x1.2475d5a90db84p-1},
    {0xfe29885da1b91, 0x1.10e20329515eep-52, 0x1.2217ca92ff7f2p-1},
    {0xfe2fb8fb54186, 0x1.12d4707310fbep-52, 0x1.1fbe0a9929620p-1},
    {0xfe35b33558d4a, 0x1.14c3e9f8e9141p-52, 0x1.1d687fe549969p-1},
    {0xfe3b799d0002a, 0x1.16b08bfc4201ep-52, 0x1.1b171573fd111p-1},
    {0xfe410e99ead7f, 0x1.189a71a78da34p-52, 0x1.18c9b709b3c50p-1},
    {0xfe46746d47734, 0x1.1a81b51ee6d88p-52, 0x1.16805128639dap-1},
    {0xfe4bad34c095c, 0x1.1c666f8f82acbp-52, 0x1.143ad105ea99cp-1},
    {0xfe50baed29524, 0x1.1e48b93e0d42ep-52, 0x1.11f9248311f38p-1},
    {0xfe559f74ebc78, 0x1.2028a9940a09fp-52, 0x1.0fbb3a2325913p-1},
    {0xfe5a5c8e41212, 0x1.2206572c4c6e9p-52, 0x1.0d810104142a0p-1},
    {0xfe5ef3e138689, 0x1.23e1d7de9c31fp-52, 0x1.0b4a68d70d9aep-1},
    {0xfe6366fd91078, 0x1.25bb40ca96bfbp-52, 0x1.091761d995d81p-1},
    {0xfe67b75c6d578, 0x1.2792a661dd37fp-52, 0x1.06e7dccf03c36p-1},
    {0xfe6be661e11aa, 0x1.29681c719d71bp-52, 0x1.04bbcafa63f2ep-1},
    {0xfe6ff55e5f4f2, 0x1.2b3bb62b82edap-52, 0x1.02931e18b822ap-1},
    {0xfe73e5900a702, 0x1.2d0d862e1b853p-52, 0x1.006dc85b8cac4p-1},
    {0xfe77b823e9e39, 0x1.2edd9e8cba98ep-52, 0x1.fc9778c7bbda1p-2},
    {0xfe7b6e37070a2, 0x1.30ac10d6e48d7p-52, 0x1.f859da7a900cap-2},
    {0xfe7f08d774243, 0x1.3278ee1f4b930p-52, 0x1.f4229cb2f7af3p-2},
    {0xfe8289053f08c, 0x1.3444470265ea1p-52, 0x1.eff1a717e8f95p-2},
    {0xfe85efb35173a, 0x1.360e2baca52d5p-52, 0x1.ebc6e20bd1f54p-2},
    {0xfe893dc840864, 0x1.37d6abe05586ap-52, 0x1.e7a236a4ec3c5p-2},
    {0xfe8c741f0cebc, 0x1.399dd6fb2b264p-52, 0x1.e3838ea5f9b85p-2},
    {0xfe8f9387d4ef6, 0x1.3b63bbfb83d03p-52, 0x1.df6ad47763a09p-2},
    {0xfe929cc879b1d, 0x1.3d28698561de0p-52, 0x1.db57f320b56b1p-2},
    {0xfe95909d388ea, 0x1.3eebede725a83p-52, 0x1.d74ad6426de33p-2},
    {0xfe986fb939aa2, 0x1.40ae571e09e74p-52, 0x1.d3436a1021080p-2},
    {0xfe9b3ac714866, 0x1.426fb2da6745dp-52, 0x1.cf419b4ae5b6dp-2},
    {0xfe9df2694b6d5, 0x1.44300e83c30a4p-52, 0x1.cb45573c0a848p-2},
    {0xfea0973abe67c, 0x1.45ef773cac75dp-52, 0x1.c74e8bb00d7c7p-2},
    {0xfea329cf166a4, 0x1.47adf9e66c336p-52, 0x1.c35d26f1d2cb8p-2},
    {0xfea5aab32952c, 0x1.496ba32488f2fp-52, 0x1.bf7117c616a17p-2},
    {0xfea81a6d5741a, 0x1.4b287f602415dp-52, 0x1.bb8a4d6716d91p-2},
    {0xfeaa797de1cf0, 0x1.4ce49acb311dcp-52, 0x1.b7a8b7807131bp-2},
    {0xfeacc85f3d920, 0x1.4ea001638a605p-52, 0x1.b3cc462b331cap-2},
    {0xfeaf07865e63c, 0x1.505abef5e5562p-52, 0x1.aff4e9ea18552p-2},
    {0xfeb13762fec13, 0x1.5214df20a8b5ap-52, 0x1.ac2293a5f5a9ep-2},
    {0xfeb3585fe2a4a, 0x1.53ce6d56a664fp-52, 0x1.a85534aa4d880p-2},
    {0xfeb56ae3162b4, 0x1.558774e1bb2c8p-52, 0x1.a48cbea20c04dp-2},
    {0xfeb76f4e284fa, 0x1.574000e555f78p-52, 0x1.a0c923946843ep-2},
    {0xfeb965fe62014, 0x1.58f81c60e8514p-52, 0x1.9d0a55e1e93dfp-2},
    {0xfebb4f4cf9d7c, 0x1.5aafd23241b59p-52, 0x1.995048418c0c6p-2},
    {0xfebd2b8f449d0, 0x1.5c672d17d733dp-52, 0x1.959aedbe09f93p-2},
    {0xfebefb16e2e3e, 0x1.5e1e37b2f8cd3p-52, 0x1.91ea39b33cb17p-2},
    {0xfec0be31ebde8, 0x1.5fd4fc89f5e38p-52, 0x1.8e3e1fcb9f115p-2},
    {0xfec2752b15a15, 0x1.618b860a31fc3p-52, 0x1.8a9693fde9188p-2},
    {0xfec42049dafd3, 0x1.6341de8a2b0a2p-52, 0x1.86f38a8ac5ab6p-2},
    {0xfec5bfd29f196, 0x1.64f8104b7260bp-52, 0x1.8354f7faa0dd9p-2},
    {0xfec75406ceef4, 0x1.66ae257c99672p-52, 0x1.7fbad11b8d911p-2},
    {0xfec8dd2500cb4, 0x1.6864283b13137p-52, 0x1.7c250aff414b0p-2},
    {0xfeca5b6911f12, 0x1.6a1a22950b2b1p-52, 0x1.78939af9252ebp-2},
    {0xfecbcf0c427fe, 0x1.6bd01e8b343bbp-52, 0x1.7506769c7b1edp-2},
    {0xfecd38454fb15, 0x1.6d8626128d352p-52, 0x1.717d93ba9614cp-2},
    {0xfece97488c8b3, 0x1.6f3c43161f854p-52, 0x1.6df8e86124caap-2},
    {0xfecfec47f91b7, 0x1.70f27f78b68ebp-52, 0x1.6a786ad88de21p-2},
    {0xfed1377358528, 0x1.72a8e516914c6p-52, 0x1.66fc11a25cbe2p-2},
    {0xfed278f844903, 0x1.745f7dc70eedcp-52, 0x1.6383d377be515p-2},
    {0xfed3b10242f4c, 0x1.7616535e5731fp-52, 0x1.600fa7480d2c8p-2},
    {0xfed4dfbad586e, 0x1.77cd6faeff449p-52, 0x1.5c9f84376c244p-2},
    {0xfed605498c3dd, 0x1.7984dc8babd93p-52, 0x1.5933619d6eebep-2},
    {0xfed721d414fe8, 0x1.7b3ca3c8b1409p-52, 0x1.55cb3703d0100p-2},
    {0xfed8357e4a982, 0x1.7cf4cf3db22fbp-52, 0x1.5266fc2533bedp-2},
    {0xfed9406a42cc8, 0x1.7ead68c73dee7p-52, 0x1.4f06a8ebf6d92p-2},
    {0xfeda42b85b704, 0x1.80667a486ea1fp-52, 0x1.4baa357109ca2p-2},
    {0xfedb3c8746ab4, 0x1.82200dac88676p-52, 0x1.485199fad6ad4p-2},
    {0xfedc2df416652, 0x1.83da2ce899f15p-52, 0x1.44fccefc324fep-2},
    {0xfedd171a46e52, 0x1.8594e1fd1f5bdp-52, 0x1.41abcd1357a19p-2},
    {0xfeddf813c8ad3, 0x1.875036f7a7ec5p-52, 0x1.3e5e8d08ed2dbp-2},
    {0xfeded0f909980, 0x1.890c35f47f72dp-52, 0x1.3b1507cf143aep-2},
    {0xfedfa1e0fd414, 0x1.8ac8e9205c043p-52, 0x1.37cf368081379p-2},
    {0xfee06ae124bc4, 0x1.8c865aba10c9cp-52, 0x1.348d125f9d19ep-2},
    {0xfee12c0d95a06, 0x1.8e44951446a27p-52, 0x1.314e94d5af62fp-2},
    {0xfee1e579006e0, 0x1.9003a2973b58fp-52, 0x1.2e13b77210766p-2},
    {0xfee29734b6524, 0x1.91c38dc288347p-52, 0x1.2adc73e963fddp-2},
    {0xfee34150ae4bc, 0x1.9384612ef0afcp-52, 0x1.27a8c414db11ep-2},
    {0xfee3e3db89b3c, 0x1.954627903a28ap-52, 0x1.2478a1f17de89p-2},
    {0xfee47ee2982f4, 0x1.9708ebb70d5eep-52, 0x1.214c079f7cc9ep-2},
    {0xfee51271db086, 0x1.98ccb892e2a31p-52, 0x1.1e22ef6188116p-2},
    {0xfee59e9407f41, 0x1.9a919933f99bfp-52, 0x1.1afd539c2f050p-2},
    {0xfee623528b42e, 0x1.9c5798cd5d92cp-52, 0x1.17db2ed5454e8p-2},
    {0xfee6a0b5897f1, 0x1.9e1ec2b6f7411p-52, 0x1.14bc7bb34ee67p-2},
    {0xfee716c3e077a, 0x1.9fe7226fad24ap-52, 0x1.11a134fcf2423p-2},
    {0xfee7858327b82, 0x1.a1b0c39f93692p-52, 0x1.0e895598709c4p-2},
    {0xfee7ecf7b06ba, 0x1.a37bb21a2c85bp-52, 0x1.0b74d88b242dap-2},
    {0xfee84d2484ab2, 0x1.a547f9e0bbb88p-52, 0x1.0863b8f904336p-2},
    {0xfee8a60b66343, 0x1.a715a724aa9a4p-52, 0x1.0555f2242e9d9p-2},
    {0xfee8f7accc851, 0x1.a8e4c64a0313dp-52, 0x1.024b7f6c7747ep-2},
    {0xfee94207e25da, 0x1.aab563e9ff108p-52, 0x1.fe88b89df93c5p-3},
    {0xfee9851a829ea, 0x1.ac878cd5af5cep-52, 0x1.f88108cb83235p-3},
    {0xfee9c0e13485c, 0x1.ae5b4e18bb336p-52, 0x1.f27fe6ce998d2p-3},
    {0xfee9f557273f4, 0x1.b030b4fc3a11ap-52, 0x1.ec854a4c99c44p-3},
    {0xfeea22762ccae, 0x1.b207cf09a985bp-52, 0x1.e6912b2283cddp-3},
    {0xfeea4836b42ac, 0x1.b3e0aa0e00c00p-52, 0x1.e0a3816457184p-3},
    {0xfeea668fc2d71, 0x1.b5bb541ce3d03p-52, 0x1.dabc455c7900ap-3},
    {0xfeea7d76ed6fa, 0x1.b797db93f8927p-52, 0x1.d4db6f8b2514fp-3},
    {0xfeea8ce04fa0a, 0x1.b9764f1e5f73cp-52, 0x1.cf00f8a5e6fccp-3},
    {0xfeea94be8333b, 0x1.bb56bdb85256ep-52, 0x1.c92cd9971df53p-3},
    {0xfeea950296410, 0x1.bd3936b2ec0a2p-52, 0x1.c35f0b7d89d47p-3},
    {0xfeea8d9c0075e, 0x1.bf1dc9b81ae83p-52, 0x1.bd9787abe18a1p-3},
    {0xfeea7e7897654, 0x1.c10486cec16a0p-52, 0x1.b7d647a8731aap-3},
    {0xfeea678481d24, 0x1.c2ed7e5f07a2dp-52, 0x1.b21b452ccd13ap-3},
    {0xfeea48aa29e83, 0x1.c4d8c136e0d1cp-52, 0x1.ac667a2571807p-3},
    {0xfeea21d22e4da, 0x1.c6c6608ec8705p-52, 0x1.a6b7e0b19267ep-3},
    {0xfee9f2e352024, 0x1.c8b66e0eba617p-52, 0x1.a10f7322d7e3dp-3},
    {0xfee9bbc26af2e, 0x1.caa8fbd36a2abp-52, 0x1.9b6d2bfd2fe5ap-3},
    {0xfee97c524f2e4, 0x1.cc9e1c73bd690p-52, 0x1.95d105f6a7c27p-3},
    {0xfee93473c0a3a, 0x1.ce95e3068e037p-52, 0x1.903afbf74fa69p-3},
    {0xfee8e40557516, 0x1.d0906328b8f6ep-52, 0x1.8aab09192815bp-3},
    {0xfee88ae369c7a, 0x1.d28db1037ef20p-52, 0x1.852128a819a38p-3},
    {0xfee828e7f3dfd, 0x1.d48de1533c647p-52, 0x1.7f9d5621f7175p-3},
    {0xfee7bdea7b888, 0x1.d691096e7f123p-52, 0x1.7a1f8d368a323p-3},
    {0xfee749bff37ff, 0x1.d8973f4d7fba5p-52, 0x1.74a7c9c7ab5a6p-3},
    {0xfee6cc3a9bd5e, 0x1.daa0999206e70p-52, 0x1.6f3607e964716p-3},
    {0xfee64529e007e, 0x1.dcad2f8fc490ep-52, 0x1.69ca43e21f25cp-3},
    {0xfee5b45a32888, 0x1.debd195522e37p-52, 0x1.64647a2adf19cp-3},
    {0xfee51994e57b6, 0x1.e0d06fb49d21cp-52, 0x1.5f04a76f883f9p-3},
    {0xfee474a0006cf, 0x1.e2e74c4ea46f6p-52, 0x1.59aac88f31d6cp-3},
    {0xfee3c53e12c50, 0x1.e501c99c1d188p-52, 0x1.5456da9c86835p-3},
    {0xfee30b2e02ad8, 0x1.e72002f97fe25p-52, 0x1.4f08dade31fc1p-3},
    {0xfee2462ad8205, 0x1.e94214b2abf0ap-52, 0x1.49c0c6cf5ce2dp-3},
    {0xfee175eb83c5a, 0x1.eb681c0f76f08p-52, 0x1.447e9c20375d5p-3},
    {0xfee09a22a1447, 0x1.ed9237610a73ap-52, 0x1.3f4258b6931aep-3},
    {0xfedfb27e349cc, 0x1.efc086101eca9p-52, 0x1.3a0bfaae8d7eep-3},
    {0xfedebea76216c, 0x1.f1f328ac25321p-52, 0x1.34db805b4ab88p-3},
    {0xfeddbe422047e, 0x1.f42a40fb74d6dp-52, 0x1.2fb0e847c2a65p-3},
    {0xfedcb0ece39d3, 0x1.f665f20c90168p-52, 0x1.2a8c3137a071ap-3},
    {0xfedb964042cf4, 0x1.f8a6604899782p-52, 0x1.256d5a2835eb7p-3},
    {0xfeda6dce938c9, 0x1.faebb187122bfp-52, 0x1.2054625183c34p-3},
    {0xfed937237e98d, 0x1.fd360d22fe785p-52, 0x1.1b41492757d42p-3},
    {0xfed7f1c38a836, 0x1.ff859c118f60bp-52, 0x1.16340e5a82d63p-3},
    {0xfed69d2b9c02b, 0x1.00ed447d3a075p-51, 0x1.112cb1da26eb9p-3},
    {0xfed538d06ae00, 0x1.021a8028fc947p-51, 0x1.0c2b33d5209bap-3},
    {0xfed3c41dea422, 0x1.034a983a902abp-51, 0x1.072f94bb8bf85p-3},
    {0xfed23e76a2fd8, 0x1.047da4e3ef5c7p-51, 0x1.0239d54067d2ap-3},
    {0xfed0a732fe644, 0x1.05b3bf6adb37ep-51, 0x1.fa93ecb6b222cp-4},
    {0xfecefda07fe34, 0x1.06ed023a72668p-51, 0x1.f0bff29520e1cp-4},
    {0xfecd4100eb7b8, 0x1.082988f632e17p-51, 0x1.e6f7bf29aa54bp-4},
    {0xfecb708956eb4, 0x1.0969708e8a254p-51, 0x1.dd3b56176e88fp-4},
    {0xfec98b61230c1, 0x1.0aacd7571c0c4p-51, 0x1.d38abb9bd91e5p-4},
    {0xfec790a0da978, 0x1.0bf3dd1eed448p-51, 0x1.c9e5f493b740ap-4},
    {0xfec57f50f31fe, 0x1.0d3ea34aa3d30p-51, 0x1.c04d0680b1015p-4},
    {0xfec356686c962, 0x1.0e8d4cf116593p-51, 0x1.b6bff78f2e233p-4},
    {0xfec114cb4b335, 0x1.0fdffefa69fb6p-51, 0x1.ad3ece9caf633p-4},
    {0xfebeb948e6fd0, 0x1.1136e04207041p-51, 0x1.a3c9933ea6286p-4},
    {0xfebc429a0b692, 0x1.129219bbb5d35p-51, 0x1.9a604dc9d5b19p-4},
    {0xfeb9af5ee0cdc, 0x1.13f1d69c4096dp-51, 0x1.9103075a4a0abp-4},
    {0xfeb6fe1c98542, 0x1.1556448602e3bp-51, 0x1.87b1c9dbf2852p-4},
    {0xfeb42d3ad1f9e, 0x1.16bf93b9deef3p-51, 0x1.7e6ca013eefd6p-4},
    {0xfeb13b00b2d4b, 0x1.182df74d21261p-51, 0x1.753395aaa1176p-4},
    {0xfeae2591a02e9, 0x1.19a1a564eebacp-51, 0x1.6c06b73694a4cp-4},
    {0xfeaaeae992257, 0x1.1b1ad777f2f8ep-51, 0x1.62e6124854d18p-4},
    {0xfea788d8ee326, 0x1.1c99ca971a694p-51, 0x1.59d1b577466a4p-4},
    {0xfea3fcffd73e5, 0x1.1e1ebfbe4ae39p-51, 0x1.50c9b06fa2baep-4},
    {0xfea044c8dd9f6, 0x1.1fa9fc2e2d901p-51, 0x1.47ce1401b2213p-4},
    {0xfe9c5d62f563b, 0x1.213bc9d04cc81p-51, 0x1.3edef23269a86p-4},
    {0xfe9843ba947a4, 0x1.22d477a6fd3eep-51, 0x1.35fc5e4d93e70p-4},
    {0xfe93f471d4728, 0x1.24745a4ac9c24p-51, 0x1.2d266cf9b3111p-4},
    {0xfe8f6bd76c5d6, 0x1.261bcc77658e0p-51, 0x1.245d344dd0d91p-4},
    {0xfe8aa5dc4e8e6, 0x1.27cb2faa8592ep-51, 0x1.1ba0cbe97897dp-4},
    {0xfe859e07ab1ea, 0x1.2982ecd770e78p-51, 0x1.12f14d0f2179dp-4},
    {0xfe804f690a940, 0x1.2b437532a0a52p-51, 0x1.0a4ed2c159625p-4},
    {0xfe7ab488233c0, 0x1.2d0d43196db97p-51, 0x1.01b979e30e497p-4},
    {0xfe74c751f6aa5, 0x1.2ee0db1a978f5p-51, 0x1.f262c2b6c6e35p-5},
    {0xfe6e8102aa202, 0x1.30becd256aeeep-51, 0x1.e16d547b25181p-5},
    {0xfe67da0b6abd8, 0x1.32a7b5e68a4a3p-51, 0x1.d092efeadf162p-5},
    {0xfe60c9f38307e, 0x1.349c405ae12a3p-51, 0x1.bfd3e0f282a2cp-5},
    {0xfe5947338f742, 0x1.369d27a33a840p-51, 0x1.af30790385f70p-5},
    {0xfe51470977280, 0x1.38ab39256410ap-51, 0x1.9ea90f9295563p-5},
    {0xfe48bd436f458, 0x1.3ac7570ae88fap-51, 0x1.8e3e02a68b5abp-5},
    {0xfe3f9bffd1e37, 0x1.3cf27b31704a6p-51, 0x1.7defb77af271ep-5},
    {0xfe35d35eeb19c, 0x1.3f2dbaa60f475p-51, 0x1.6dbe9b398d064p-5},
    {0xfe2b5122fe4fe, 0x1.417a49cb9e5dap-51, 0x1.5dab23cf2add4p-5},
    {0xfe20003995557, 0x1.43d9815545e94p-51, 0x1.4db5d0e11275dp-5},
    {0xfe13c82788314, 0x1.464ce44a73a15p-51, 0x1.3ddf2ce98eecbp-5},
    {0xfe068c4ee67b0, 0x1.48d62759c43bcp-51, 0x1.2e27ce83df497p-5},
    {0xfdf82b02b71aa, 0x1.4b7739d6b5a27p-51, 0x1.1e9059f1f6abcp-5},
    {0xfde87c57efeaa, 0x1.4e3250dcd8902p-51, 0x1.0f1982e968011p-5},
    {0xfdd7509c63bfd, 0x1.5109f53e9ac41p-51, 0x1.ff881d718a5c4p-6},
    {0xfdc46e529bf13, 0x1.54011523a7e42p-51, 0x1.e121adb828c75p-6},
    {0xfdaf8f82e0282, 0x1.571b1a94ae41bp-51, 0x1.c301983cd091ap-6},
    {0xfd985e1b2ba75, 0x1.5a5c08b718dd9p-51, 0x1.a529f4e22ebf8p-6},
    {0xfd7e6ef48cf04, 0x1.5dc8a243ad0fep-51, 0x1.879d1b600c10ap-6},
    {0xfd613adbd650b, 0x1.61669cf861e4cp-51, 0x1.6a5daf40bbf82p-6},
    {0xfd40149e2f012, 0x1.653ce7b006aeap-51, 0x1.4d6eaf2fbb064p-6},
    {0xfd1a1a7b4c7ac, 0x1.69540be9fe5c3p-51, 0x1.30d388dab5e13p-6},
    {0xfcee204761f9e, 0x1.6db6b8d09e232p-51, 0x1.1490334603012p-6},
    {0xfcba8d85e11b2, 0x1.72728f05f7a34p-51, 0x1.f152a4f72dd49p-7},
    {0xfc7d26ecd2d22, 0x1.7799556090673p-51, 0x1.ba48d274f8facp-7},
    {0xfc32b2f1e22ed, 0x1.7d42df4d6ce8cp-51, 0x1.841040d8da478p-7},
    {0xfbd6581c0b83a, 0x1.839030529f234p-51, 0x1.4eb96421acfe0p-7},
    {0xfb606c4005434, 0x1.8ab0fbfaa7c14p-51, 0x1.1a59229952f92p-7},
    {0xfac40582a2874, 0x1.92ee0946f4496p-51, 0x1.ce160f8ec6837p-8},
    {0xf9e971e014598, 0x1.9cbee014057abp-51, 0x1.69ea8d90cb85dp-8},
    {0xf89fa48a41dfc, 0x1.a8fdc7894775ap-51, 0x1.08a1f03b0b1fdp-8},
    {0xf66c5f7f0302c, 0x1.b981f3878fdb1p-51, 0x1.55f9f43c1b067p-9},
    {0xf1a5a4b331c4a, 0x1.d3bb48209ad33p-51, 0x1.4a605b6b9f70fp-10},
};

static Pcg64 syn_rng(json_object *spec, const char *name) {
    const char *seed = json_object_to_json_string_ext(jget(spec, "seed"), JSON_C_TO_STRING_PLAIN);
    char text[256], hash[65];
    int n = snprintf(text, sizeof(text), "%s:%s", seed, name);
    need(n > 0 && n < (int)sizeof(text), "synthetic stream name too long");
    sha256_hex(text, n, hash);
    u64 key = 0;
    for (int i = 14; i >= 0; i -= 2) {
        char digits[] = {hash[i], hash[i + 1], 0};
        key = (key << 8) | strtoul(digits, NULL, 16);
    }
    return pcg64(key);
}

static double syn_normal(Pcg64 *rng) {
    for (;;) {
        u64 bits = pcg64_next(rng);
        unsigned idx = bits & 255;
        u64 draw = (bits >> 9) & 0x000fffffffffffffull;
        double value = draw * syn_ziggurat[idx].width;
        if (bits & 256) {
            value = -value;
        }
        if (draw < syn_ziggurat[idx].threshold) {
            return value;
        }
        if (!idx) {
            for (;;) {
                double x = -0.273661237329758272 * log1p(-pcg64_double(rng));
                double y = -log1p(-pcg64_double(rng));
                if (y + y > x * x) {
                    double tail = 3.6541528853610088 + x;
                    return draw & 256 ? -tail : tail;
                }
            }
        }
        double pdf = syn_ziggurat[idx].pdf;
        if ((syn_ziggurat[idx - 1].pdf - pdf) * pcg64_double(rng) + pdf <
            exp(-0.5 * value * value)) {
            return value;
        }
    }
}

static double syn_sum(const double *x, size_t n) {
    if (n < 8) {
        double sum = -0.0;
        for (size_t i = 0; i < n; i++) {
            sum += x[i];
        }
        return sum;
    }
    if (n <= 128) {
        double partial[8];
        memcpy(partial, x, sizeof(partial));
        size_t i = 8;
        for (; i + 8 <= n; i += 8) {
            for (int k = 0; k < 8; k++) {
                partial[k] += x[i + k];
            }
        }
        double sum = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
            ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for (; i < n; i++) {
            sum += x[i];
        }
        return sum;
    }
    size_t half = (n / 2) & ~(size_t)7;
    return syn_sum(x, half) + syn_sum(x + half, n - half);
}

static void syn_center(double *field, double *scratch, size_t n) {
    double mean = syn_sum(field, n) / n;
    for (size_t i = 0; i < n; i++) {
        double d = field[i] - mean;
        scratch[i] = d * d;
    }
    double std = sqrt(syn_sum(scratch, n) / n);
    if (std == 0) {
        std = 1;
    }
    for (size_t i = 0; i < n; i++) {
        field[i] = (field[i] - mean) / std;
    }
}

static void syn_filter_order(Arena *a, const double *src, double *dst, size_t n, size_t step,
    int axis, double sigma, double truncate, int wrap, int order) {
    int radius = (int)(truncate * sigma + 0.5);
    if (!radius) {
        memcpy(dst, src, n * sizeof(*dst));
        return;
    }
    size_t mark = a->used;
    double *kernel = push(a, (2 * (size_t)radius + 1) * sizeof(*kernel));
    double scale = -0.5 / (sigma * sigma);
    for (int d = -radius; d <= radius; d++) {
        kernel[d + radius] = exp(scale * d * d);
    }
    double total = syn_sum(kernel, 2 * (size_t)radius + 1);
    double variance = sigma * sigma;
    for (int d = -radius; d <= radius; d++) {
        kernel[d + radius] /= total;
        if (order == 1) {
            kernel[d + radius] *= -d / variance;
        } else if (order == 2) {
            kernel[d + radius] *= (d * d / variance - 1) / variance;
        }
    }
    int period = wrap ? axis : 2 * axis;
    size_t span = (size_t)axis * step;
    int *positions = push(a, (size_t)axis * (2 * (size_t)radius + 1) * sizeof(*positions));
    for (int coord = 0; coord < axis; coord++) {
        int *index = positions + (size_t)coord * (2 * (size_t)radius + 1);
        for (int d = -radius; d <= radius; d++) {
            int at = (coord + d) % period;
            if (at < 0) {
                at += period;
            }
            index[d + radius] = at < axis ? at : period - 1 - at;
        }
    }
    for (size_t block = 0; block < n; block += span) {
        for (int coord = 0; coord < axis; coord++) {
            size_t center = block + (size_t)coord * step;
            const int *index = positions + (size_t)coord * (2 * (size_t)radius + 1);
            for (size_t offset = 0; offset < step; offset++) {
                double value = kernel[radius] * src[center + offset];
                for (int d = -radius; d < 0; d++) {
                    size_t left = block + (size_t)index[d + radius] * step + offset;
                    size_t right = block + (size_t)index[radius - d] * step + offset;
                    double pair = order == 1 ? src[right] - src[left] : src[left] + src[right];
                    value += pair * kernel[d + radius];
                }
                dst[center + offset] = value;
            }
        }
    }
    a->used = mark;
}

static void syn_filter(Arena *a, const double *src, double *dst, size_t n, size_t step, int axis,
    double sigma, double truncate, int wrap) {
    syn_filter_order(a, src, dst, n, step, axis, sigma, truncate, wrap, 0);
}

static double syn_quantile(double probability) {
    if (probability <= 0 || probability >= 1) {
        return probability == 0 ? -INFINITY : probability == 1 ? INFINITY : NAN;
    }
    double p = fmin(probability, 1 - probability), target = 2 * p;
    double lo = 0, hi = 40;
    for (int k = 0; k < 58; k++) {
        double mid = (lo + hi) * 0.5;
        if (erfc(mid * M_SQRT1_2) > target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return copysign((lo + hi) * 0.5, probability - 0.5);
}

static void syn_field(Arena *a, Pcg64 *rng, double *out, double *scratch, int dims,
    const int *shape, const double *sigma, double truncate, int wrap) {
    size_t n = 1;
    for (int d = 0; d < dims; d++) {
        n *= shape[d];
    }
    for (size_t i = 0; i < n; i++) {
        out[i] = syn_normal(rng);
    }
    double *from = out, *to = scratch;
    size_t stride = n;
    for (int d = 0; d < dims; d++) {
        stride /= shape[d];
        if (sigma[d] > 0) {
            syn_filter(a, from, to, n, stride, shape[d], sigma[d], truncate, wrap);
            double *swap = from;
            from = to;
            to = swap;
        }
    }
    if (from != out) {
        memcpy(out, from, n * sizeof(*out));
    }
}
