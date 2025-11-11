import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.tsp_construct_set import TSPSEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eohs import EoHS,EoHSProfiler


def main():

    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='deepseek-v3',  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = TSPSEvaluation(
        timeout_seconds=120,
        datasets=['./dataset_tsp_3_0.03_32_training.pkl', './dataset_tsp_3_0.07_32_training.pkl',
                 './dataset_tsp_10_0.03_32_training.pkl', './dataset_tsp_10_0.07_32_training.pkl'],
        return_list=True)

    method = EoHS(llm=llm,
                 profiler=EoHSProfiler(log_dir='logs/eohs', log_style='simple'),
                 evaluation=task,
                 max_sample_nums=2000,
                 max_generations=1000,
                 pop_size=10,
                 num_samplers=4,
                 num_evaluators=4,
                 debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
