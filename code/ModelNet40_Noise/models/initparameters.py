def init_parameters_test():
    PermutationEquivariantLayer_parameters = {"permutation_equivariant_layers_nums": 1,
                                              "permutation_equivariant_layers_parameters": [{"dimension": [12, 64],
                                                                                             "perm_op": ["max", None]}
                                                                                            ],
                               "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                               "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                                              }

    RationalHatLayer_parameters = {"rational_hat_layers_nums": 1,
                                   "rational_hat_layers_parameters": [{"p": 2,
                                                                       "lnum": 128}
                                                                      ],
                               "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                               "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                                   }

    ExponentialLayer_parameters = {"exponential_layers_nums": 1,
                                   "exponential_layers_parameters": [{"lnum": 128}
                                                                      ],
                               "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                               "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                                   }

    RationalLayer_parameters = {"rational_layers_nums": 1,
                                "rational_layers_parameters": [{"lnum": 128}
                                                                ],
                               "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                               "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                                }

    LandscapeLayer_parameters = {"landscape_layers_nums": 1,
                                 "landscape_layers_parameters": [{"lsample_num": 128}
                                                                ],
                               "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                               "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                                 }

    BettiLayer_parameters = {"betti_layers_nums": 1,
                             "betti_layers_parameters": [{"lsample_num": 128,
                                                          "theta": 10}
                                                        ],
                             "weight_layers_parameters": [{"pweight": "power",
                                                           "w_power": 2,
                                                           "w_grid_shape": -1,
                                                           "w_grid_bnds": -1,
                                                           "w_guass_num": -1}
                                                            ],
                             "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                             }

    EntropyLayer_parameters = {"entropy_layers_nums": 1,
                               "entropy_layers_parameters": [{"lsample_num": 128,
                                                              "theta": 10}
                                                             ],
                               "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                               "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                               }

    ImageLayer_parameters = {"image_layers_nums": 1,
                             "image_layers_parameters": [{"image_size": (8, 8),
                                                          "image_bnds": ((-.001, 1.001), (-.001, 1.001))}
                                                        ],
                             "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                             "op_parameters": {"h1_op": [["max"]],
                                                 "h2_op": [["max"]]}
                             }

    HightOrderGuasssianKernel_parameters = {"hight_order_guassian_kernel_layers_nums": 1,
                                            "hight_order_guassian_kernel_layers_parameters": [{"dim_feature": 128,
                                                                                                "p1": 2,
                                                                                                "p2": 2}
                                                                                              ],
                                            "weight_layers_parameters": [{"pweight": "gmix",
                                                                          "w_power": 2,
                                                                          "w_grid_shape": -1,
                                                                          "w_grid_bnds": -1,
                                                                          "w_guass_num": 3}
                                                                         ],
                                            "op_parameters": {"h1_op": [["max"]],
                                                              "h2_op": [["max"]]}
                                            }

    GLPKernel_parameters = {"glp_kernel_layers_nums": 1,
                            "glp_kernel_layers_parameters": [{"dim_feature": 128}
                                                            ],
                            "weight_layers_parameters": [{"pweight": "power",
                                                             "w_power": 2,
                                                             "w_grid_shape": -1,
                                                             "w_grid_bnds": -1,
                                                             "w_guass_num": -1}
                                                            ],
                            "op_parameters": {"h1_op": [["max"]],
                                              "h2_op": [["max"]]}
                            }
    return [PermutationEquivariantLayer_parameters, RationalHatLayer_parameters, ExponentialLayer_parameters, \
           RationalLayer_parameters, LandscapeLayer_parameters, BettiLayer_parameters, EntropyLayer_parameters, \
           ImageLayer_parameters, HightOrderGuasssianKernel_parameters, GLPKernel_parameters]


def cal_topofeature_dim(PermutationEquivariantLayer_parameters, RationalHatLayer_parameters, ExponentialLayer_parameters, \
        RationalLayer_parameters, LandscapeLayer_parameters, BettiLayer_parameters, EntropyLayer_parameters, \
        ImageLayer_parameters, HightOrderGuasssianKernel_parameters, GLPKernel_parameters, *args):
    dim = 0

    if PermutationEquivariantLayer_parameters != None:
        for i in range(PermutationEquivariantLayer_parameters["permutation_equivariant_layers_nums"]):
            dim = dim + 2 * PermutationEquivariantLayer_parameters["permutation_equivariant_layers_parameters"][i]["dimension"][-1]

    if RationalHatLayer_parameters != None:
        for i in range(RationalHatLayer_parameters["rational_hat_layers_nums"]):
            dim = dim + 2 * RationalHatLayer_parameters["rational_hat_layers_parameters"][i]["lnum"]

    if ExponentialLayer_parameters != None:
        for i in range(ExponentialLayer_parameters["exponential_layers_nums"]):
            dim = dim + 2 * ExponentialLayer_parameters["exponential_layers_parameters"][i]["lnum"]

    if RationalLayer_parameters != None:
        for i in range(RationalLayer_parameters["rational_layers_nums"]):
            dim = dim + 2 * RationalLayer_parameters["rational_layers_parameters"][i]["lnum"]

    if LandscapeLayer_parameters != None:
        for i in range(LandscapeLayer_parameters["landscape_layers_nums"]):
            dim = dim + 2 * LandscapeLayer_parameters["landscape_layers_parameters"][i]["lsample_num"]

    if BettiLayer_parameters != None:
        for i in range(BettiLayer_parameters["betti_layers_nums"]):
            dim = dim + 2 * BettiLayer_parameters["betti_layers_parameters"][i]["lsample_num"]

    if EntropyLayer_parameters != None:
        for i in range(EntropyLayer_parameters["entropy_layers_nums"]):
            dim = dim + 2 * EntropyLayer_parameters["entropy_layers_parameters"][i]["lsample_num"]

    if ImageLayer_parameters != None:
        for i in range(ImageLayer_parameters["image_layers_nums"]):
            dim = dim + 2 * ImageLayer_parameters["image_layers_parameters"][i]["image_size"][0] * ImageLayer_parameters["image_layers_parameters"][i]["image_size"][1]

    if HightOrderGuasssianKernel_parameters != None:
        for i in range(HightOrderGuasssianKernel_parameters["hight_order_guassian_kernel_layers_nums"]):
            dim = dim + 2 * HightOrderGuasssianKernel_parameters["hight_order_guassian_kernel_layers_parameters"][i]["dim_feature"]

    if GLPKernel_parameters != None:
        for i in range(GLPKernel_parameters["glp_kernel_layers_nums"]):
            dim = dim + 2 * GLPKernel_parameters["glp_kernel_layers_parameters"][i]["dim_feature"]

    return dim