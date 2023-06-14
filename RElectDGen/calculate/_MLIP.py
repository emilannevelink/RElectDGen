from nequip.train import Trainer

def nn_from_results(root='results',train_directory=None,template=''):
    # spack cannot load nequip due to older ase version
    import torch
    from nequip.utils import Config
    from nequip.ase.nequip_calculator import NequIPCalculator
    from nequip.data.transforms import TypeMapper
    from RElectDGen.utils.save import get_results_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train_directory is None:
        train_directory = get_results_dir(root,template)
    print(train_directory,flush=True)
    # file_config = train_directory + "/config_final.yaml"
    # MLP_config = Config.from_file(file_config)

    # model_path = train_directory + "/best_model.pth"
    
    # model_state_dict = torch.load(model_path, map_location=torch.device(device))
    # try:
    #     dataset = dataset_from_config(MLP_config)
    #     model = model_from_config(
    #             config=MLP_config, initialize=True, dataset=dataset
    #         )
    # except Exception as e:
    #     print('Model not initialized')
    #     print(e)
    #     model = model_from_config(
    #             config=MLP_config, initialize=False
    #         )
    # if not isinstance(model_state_dict, OrderedDict):
    #     model_state_dict = model_state_dict.state_dict() # for backwards compatability
    # model.load_state_dict(model_state_dict)
    

    model, _ = Trainer.load_model_from_training_session(
        traindir=train_directory
    )
    MLP_config = Config.from_file(train_directory + "/config_final.yaml")
    
    model.eval()
    model.to(torch.device(device))
    chemical_symbol_to_type = MLP_config.get('chemical_symbol_to_type')
    transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)

    # if MLP_config.compile_model:
    import e3nn
    model = e3nn.util.jit.compile(model)
    print('compiled model', flush=True)
    torch.jit._set_fusion_strategy([('DYNAMIC',MLP_config.get("_jit_bailout_depth",2))])
    # torch._C._jit_set_profiling_executor(False)
    # model = torch.jit.script(model)
    # model = torch.jit.freeze(model)
    # model = torch.jit.optimize_for_inference(model)
    
    calc_nn = NequIPCalculator(model=model, r_max=MLP_config.r_max,device=device, transform=transform)

    return calc_nn, model, MLP_config

def nns_from_results(root='results',n_ensemble=4,template=''):
    from RElectDGen.utils.save import check_nan_parameters
    model = []
    MLP_config = []
    for i in range(n_ensemble):
        rooti = root + f'_{i}'
        try:
            calc_tmp, mod, conf = nn_from_results(root=rooti,template=template)
            training_success = check_nan_parameters(mod)
            if training_success:
                model.append(mod)
                MLP_config.append(conf)
                calc_nn = calc_tmp
        except UnboundLocalError as e:
            print(f'Failed to load model from {rooti}')
            print(e,flush=True)
    
    print(f'Kept {len(model)} of {n_ensemble} models', flush=True)
    return calc_nn, model, MLP_config