from __future__ import annotations

import sys

sys.path.append('../../')  # This is for finding all the modules


from llm4ad.task.optimization.online_bin_packing_set import OBPSEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.reevo import ReEvo, ReEvoProfiler

if __name__ == '__main__':

    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='deepseek-v3',  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = OBPSEvaluation(
        timeout_seconds=120,
        dataset='./dataset_100_2k_128_5_80_training.pkl',
        return_list=False)


    method = ReEvo(
        llm=llm,
        profiler=ReEvoProfiler(log_dir='logs/reevo', log_style='complex'),
        evaluation=task,
        max_sample_nums=2000,
        pop_size=10,
        num_samplers=4,
        num_evaluators=4,
        debug_mode=False,
    )
    method._evaluator._debug_mode = False
    method.run()
