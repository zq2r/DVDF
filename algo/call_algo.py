# import all algorithms this benchmark implement

def call_algo(algo_name, config, mode, device):
    if mode == 0:
       pass

    elif mode == 1:
        pass

    elif mode == 2:
        pass

    elif mode == 3:
        algo_name = algo_name.lower()
        assert algo_name in ["dv_igdf"]
        # offline offline setting
        from offline_offline.dv_igdf import DV_IGDF

        algo_to_call = {
            "dv_igdf": DV_IGDF,
        }

        algo = algo_to_call[algo_name]
        policy = algo(config, device)
    
    return policy